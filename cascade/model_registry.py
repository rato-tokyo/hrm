"""
CASCADEフレームワーク - モデルレジストリ

事前学習済みモデルを疎結合で管理するレジストリ。
モデルの切り替えを容易にし、実験の柔軟性を高める。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any, Callable
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PretrainedConfig


@dataclass
class ModelSpec:
    """
    モデル仕様。

    Attributes:
        name: モデル識別名
        hf_name: Hugging Face Hub上のモデル名
        dim: hidden次元
        num_layers: レイヤー数
        description: モデルの説明
        recommended_for: 推奨用途
    """
    name: str
    hf_name: str
    dim: int
    num_layers: int
    description: str = ""
    recommended_for: str = ""


# 事前定義されたモデル仕様
PRETRAINED_MODELS: Dict[str, ModelSpec] = {
    # SmolLM2シリーズ（推奨）
    # 両モデルとも同じSmolLM-Corpusで事前学習済み
    "smollm2-135m": ModelSpec(
        name="smollm2-135m",
        hf_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        dim=576,  # hidden_size
        num_layers=30,  # num_hidden_layers
        # num_attention_heads=9, num_key_value_heads=3 (GQA)
        # intermediate_size=1536, vocab_size=49152, max_position_embeddings=8192
        description="最小の対話可能モデル、高速実験向け（2Tトークン訓練）",
        recommended_for="quick_experiment",
    ),
    "smollm2-360m": ModelSpec(
        name="smollm2-360m",
        hf_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        dim=960,  # hidden_size
        num_layers=32,  # num_hidden_layers
        # num_attention_heads=15, num_key_value_heads=5 (GQA)
        # intermediate_size=2560, vocab_size=49152, max_position_embeddings=8192
        description="バランスの良い小型モデル（4Tトークン訓練）",
        recommended_for="balanced",
    ),
    "smollm2-1.7b": ModelSpec(
        name="smollm2-1.7b",
        hf_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        dim=2048,
        num_layers=24,
        description="高品質な小型モデル",
        recommended_for="quality",
    ),
    # TinyLlama
    "tinyllama-1.1b": ModelSpec(
        name="tinyllama-1.1b",
        hf_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dim=2048,
        num_layers=22,
        description="Llama2互換、3兆トークン訓練済み",
        recommended_for="llama_compatible",
    ),
    # GPT-2シリーズ（ベースライン）
    "gpt2": ModelSpec(
        name="gpt2",
        hf_name="gpt2",
        dim=768,
        num_layers=12,
        description="標準GPT-2、ベースライン用",
        recommended_for="baseline",
    ),
    "gpt2-medium": ModelSpec(
        name="gpt2-medium",
        hf_name="gpt2-medium",
        dim=1024,
        num_layers=24,
        description="GPT-2 Medium",
        recommended_for="baseline",
    ),
}


class ModelRegistry:
    """
    モデルレジストリ。

    事前学習済みモデルの管理とロードを担当。
    カスタムモデルの追加も可能。

    使用例:
        registry = ModelRegistry()

        # 事前定義モデルをロード
        model, tokenizer = registry.load("smollm2-135m")

        # カスタムモデルを登録してロード
        registry.register(ModelSpec(
            name="my-model",
            hf_name="username/my-model",
            dim=512,
            num_layers=6,
        ))
        model, tokenizer = registry.load("my-model")
    """

    def __init__(self):
        self._models: Dict[str, ModelSpec] = PRETRAINED_MODELS.copy()
        self._custom_loaders: Dict[str, Callable] = {}

    def register(self, spec: ModelSpec) -> None:
        """
        カスタムモデルを登録。

        Args:
            spec: モデル仕様
        """
        self._models[spec.name] = spec

    def register_loader(
        self,
        name: str,
        loader: Callable[[], tuple[PreTrainedModel, Any]],
    ) -> None:
        """
        カスタムローダーを登録。

        特殊なロード処理が必要な場合に使用。

        Args:
            name: モデル識別名
            loader: モデルとトークナイザを返す関数
        """
        self._custom_loaders[name] = loader

    def list_models(self) -> Dict[str, ModelSpec]:
        """登録されている全モデルを返す。"""
        return self._models.copy()

    def get_spec(self, name: str) -> ModelSpec:
        """
        モデル仕様を取得。

        Args:
            name: モデル識別名

        Returns:
            ModelSpec

        Raises:
            KeyError: モデルが見つからない場合
        """
        if name not in self._models:
            available = ", ".join(self._models.keys())
            raise KeyError(f"モデル '{name}' が見つかりません。利用可能: {available}")
        return self._models[name]

    def load(
        self,
        name: str,
        device: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
        **kwargs: Any,
    ) -> tuple[PreTrainedModel, Any]:
        """
        モデルとトークナイザをロード。

        Args:
            name: モデル識別名
            device: ロード先デバイス（"cuda", "cpu", "auto"）
            torch_dtype: データ型（torch.float16, torch.bfloat16等）
            **kwargs: AutoModelForCausalLM.from_pretrainedに渡す追加引数

        Returns:
            (model, tokenizer) タプル
        """
        # カスタムローダーがあれば使用
        if name in self._custom_loaders:
            return self._custom_loaders[name]()

        spec = self.get_spec(name)

        # ロード引数を構築
        load_kwargs: Dict[str, Any] = {}
        if device is not None:
            load_kwargs["device_map"] = device
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        load_kwargs.update(kwargs)

        # モデルとトークナイザをロード
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_name,
            **load_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)

        return model, tokenizer


# グローバルレジストリインスタンス
_default_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """デフォルトのレジストリを取得。"""
    global _default_registry
    if _default_registry is None:
        _default_registry = ModelRegistry()
    return _default_registry


def load_pretrained(
    name: str,
    device: Optional[str] = None,
    torch_dtype: Optional[Any] = None,
    **kwargs: Any,
) -> tuple[PreTrainedModel, Any]:
    """
    事前学習済みモデルをロード（ショートカット関数）。

    Args:
        name: モデル識別名（"smollm2-135m", "tinyllama-1.1b"等）
        device: ロード先デバイス
        torch_dtype: データ型
        **kwargs: 追加引数

    Returns:
        (model, tokenizer) タプル

    使用例:
        from cascade import load_pretrained, LLM

        # SmolLM2-135Mをロード
        model, tokenizer = load_pretrained("smollm2-135m")
        llm = LLM(model)

        # GPUにfloat16でロード
        model, tokenizer = load_pretrained(
            "smollm2-360m",
            device="auto",
            torch_dtype=torch.float16,
        )
    """
    return get_registry().load(name, device, torch_dtype, **kwargs)


def list_available_models() -> Dict[str, ModelSpec]:
    """
    利用可能なモデル一覧を取得。

    Returns:
        モデル名をキー、ModelSpecを値とする辞書
    """
    return get_registry().list_models()


def create_small_llm(
    dim: int,
    num_layers: int,
    vocab_size: int,
    num_heads: Optional[int] = None,
    ffn_dim: Optional[int] = None,
    max_seq_len: int = 2048,
    architecture: str = "llama",
) -> PreTrainedModel:
    """
    新規の小規模LLMを作成（未訓練）。

    後続LLMとして追加する小規模モデルを作成。
    ベースモデルと同じアーキテクチャを使用することを推奨。

    Args:
        dim: hidden次元
        num_layers: レイヤー数
        vocab_size: 語彙サイズ
        num_heads: Attentionヘッド数（デフォルト: dim // 64）
        ffn_dim: FFN次元（デフォルト: dim * 4）
        max_seq_len: 最大シーケンス長
        architecture: アーキテクチャ種別 ("llama" or "gpt2")

    Returns:
        未訓練のCausalLM

    使用例:
        # SmolLM2に合わせたLlamaアーキテクチャで作成
        base_model, _ = load_pretrained("smollm2-135m")
        new_llm = create_small_llm(
            dim=base_model.config.hidden_size,
            num_layers=2,
            vocab_size=base_model.config.vocab_size,
            architecture="llama",
        )

        # GPT-2アーキテクチャで作成
        new_llm = create_small_llm(
            dim=768,
            num_layers=4,
            vocab_size=50257,
            architecture="gpt2",
        )
    """
    if num_heads is None:
        num_heads = max(1, dim // 64)
    if ffn_dim is None:
        ffn_dim = dim * 4

    config: PretrainedConfig
    if architecture == "llama":
        from transformers import LlamaConfig

        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=dim,
            intermediate_size=ffn_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,  # MHAを使用（GQAではない）
            max_position_embeddings=max_seq_len,
            rms_norm_eps=1e-5,
            rope_theta=100000,  # SmolLM2と同じ
            hidden_act="silu",
            tie_word_embeddings=False,
        )
    elif architecture == "gpt2":
        from transformers import GPT2Config

        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=dim,
            n_head=num_heads,
            n_inner=ffn_dim,
            n_layer=num_layers,
            n_positions=max_seq_len,
        )
    else:
        raise ValueError(f"未対応のアーキテクチャ: {architecture}")

    return AutoModelForCausalLM.from_config(config)


def create_llm_from_base(
    base_model: PreTrainedModel,
    num_layers: int,
) -> PreTrainedModel:
    """
    ベースモデルと同じアーキテクチャで新規LLMを作成。

    ベースモデルのconfig（次元、語彙サイズ、アーキテクチャ）を継承し、
    指定されたレイヤー数で新しいモデルを作成。

    Args:
        base_model: ベースとなるモデル
        num_layers: 新しいモデルのレイヤー数

    Returns:
        未訓練のCausalLM（ベースモデルと同じアーキテクチャ）

    使用例:
        base_model, tokenizer = load_pretrained("smollm2-135m")
        additional_llm = create_llm_from_base(base_model, num_layers=2)
    """
    config = base_model.config

    # アーキテクチャを判定
    arch = config.architectures[0] if config.architectures else ""

    new_config: PretrainedConfig
    if "Llama" in arch:
        from transformers import LlamaConfig

        new_config = LlamaConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, "rope_theta", 10000),
            hidden_act=config.hidden_act,
            tie_word_embeddings=config.tie_word_embeddings,
        )
    elif "GPT2" in arch:
        from transformers import GPT2Config

        new_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_inner=config.n_inner,
            n_layer=num_layers,
            n_positions=config.n_positions,
        )
    else:
        raise ValueError(f"未対応のアーキテクチャ: {arch}")

    return AutoModelForCausalLM.from_config(new_config)

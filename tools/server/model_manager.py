import os

import torch
from loguru import logger

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest
from tools.server.inference import inference_wrapper as inference


class ModelManager:
    def __init__(
        self,
        mode: str,
        device: str,
        half: bool,
        compile: bool,
        llama_checkpoint_path: str,
        decoder_checkpoint_path: str,
        decoder_config_name: str,
    ) -> None:

        self.mode = mode
        self.device = device
        self.half = half
        self.compile = compile

        self.precision = torch.half if half else torch.bfloat16

        # Check if MPS or CUDA is available
        if torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("mps is available, running on mps.")
        elif not torch.cuda.is_available():
            self.device = "cpu"
            logger.info("CUDA is not available, running on CPU.")

        # Load the TTS models
        self.load_llama_model(
            llama_checkpoint_path, self.device, self.precision, self.compile, self.mode
        )
        self.load_decoder_model(
            decoder_config_name, decoder_checkpoint_path, self.device, self.precision
        )
        self.tts_inference_engine = TTSInferenceEngine(
            llama_queue=self.llama_queue,
            decoder_model=self.decoder_model,
            precision=self.precision,
            compile=self.compile,
        )

        # Warm up the models
        if self.mode == "tts":
            self.warm_up(self.tts_inference_engine)

    def load_llama_model(
        self, checkpoint_path, device, precision, compile, mode
    ) -> None:

        if mode == "tts":
            self.llama_queue = launch_thread_safe_queue(
                checkpoint_path=checkpoint_path,
                device=device,
                precision=precision,
                compile=compile,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        logger.info("LLAMA model loaded.")

    def load_decoder_model(self, config_name, checkpoint_path, device, precision=None) -> None:
        self.decoder_model = load_decoder_model(
            config_name=config_name,
            checkpoint_path=checkpoint_path,
            device=device,
            precision=precision,
        )
        # Alternate VQ decode mitigation: imagilux/miopen-conv-fix C++ extension.
        # Calls MIOpen's Immediate Mode API directly with proper workspace
        # allocation, dispatching AMD's hand-tuned conv kernels instead of
        # the workspace=0 <GemmFwdRest> fallback.
        #
        # WARNING: same-process use page-faults on Strix Halo (gfx1151) — the
        # LLM's hipMalloc/hipFree pressure leaves stale GPU page tables, and
        # the first conv call crashes with `Memory access fault by GPU node-1
        # ... Page not present or supervisor privilege.` Same bug imagilux
        # documents for gfx1201; their workaround is running the decoder in
        # a separate process with its own HIP context.
        #
        # Until we add a subprocess decoder, prefer FISH_DISABLE_MIOPEN=1
        # (PyTorch built-in GEMM-conv path, ~6x speedup, stable). This hook
        # is left in place so the patching path is one env var away if/when
        # subprocess isolation lands.
        if os.environ.get("FISH_USE_MIOPEN_CONV_FIX", "0") == "1":
            try:
                import miopen_conv_fix
                n = miopen_conv_fix.patch_module(self.decoder_model)
                logger.info(
                    f"miopen-conv-fix: patched {n} Conv1d/ConvTranspose1d layers"
                )
            except ImportError:
                logger.warning(
                    "FISH_USE_MIOPEN_CONV_FIX=1 set but miopen_conv_fix not "
                    "installed; build from "
                    "https://github.com/imagilux/miopen-conv-fix and install "
                    "with `uv pip install --no-build-isolation --no-deps ./miopen-conv-fix`"
                )
        logger.info("Decoder model loaded.")

    def warm_up(self, tts_inference_engine) -> None:
        request = ServeTTSRequest(
            text="Hello world.",
            references=[],
            reference_id=None,
            max_new_tokens=1024,
            chunk_length=200,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
            format="wav",
        )
        list(inference(request, tts_inference_engine))
        logger.info("Models warmed up.")

"""Triton version compatibility and auto-detection.

Ensures kernels work across Triton versions with graceful degradation.
"""
import torch
import warnings
from typing import Optional, Tuple


class TritonCompatibility:
    """
    Check Triton compatibility and select appropriate kernel implementations.

    Handles version differences:
      - Triton 2.0 (PyTorch 2.0): Basic support
      - Triton 2.1 (PyTorch 2.1): @triton.jit decorator changes
      - Triton 2.2+ (PyTorch 2.2+): Advanced features, tl.sort, etc.
    """

    def __init__(self):
        self.triton_version = self._get_triton_version()
        self.torch_version = self._get_torch_version()
        self.cuda_available = torch.cuda.is_available()
        self.cuda_capability = self._get_cuda_capability()

        self._print_status()

    def _get_triton_version(self) -> Optional[Tuple[int, int, int]]:
        """Get Triton version as tuple."""
        try:
            import triton
            version_str = triton.__version__
            parts = version_str.split(".")
            return tuple(int(p) for p in parts[:3])
        except (ImportError, AttributeError):
            return None

    def _get_torch_version(self) -> Tuple[int, int, int]:
        """Get PyTorch version as tuple."""
        version_str = torch.__version__.split("+")[0]
        parts = version_str.split(".")
        return tuple(int(p) for p in parts[:3])

    def _get_cuda_capability(self) -> Optional[Tuple[int, int]]:
        """Get GPU compute capability."""
        if not self.cuda_available:
            return None
        return torch.cuda.get_device_capability()

    def _print_status(self):
        """Print compatibility status."""
        print("=" * 60)
        print("Triton Compatibility Check")
        print("=" * 60)
        print(f"PyTorch: {self.torch_version}")
        print(f"Triton: {self.triton_version or 'NOT INSTALLED'}")
        print(f"CUDA: {self.cuda_available} ({self.cuda_capability or 'N/A'})")

        if self.triton_version is None:
            print("\n⚠️  Triton not installed. Kernels will fall back to PyTorch.")
            print("   Install: pip install triton")
        elif self.triton_version < (2, 0, 0):
            print("\n⚠️  Triton version too old. Some features may not work.")
            print("   Upgrade: pip install -U triton")
        else:
            print("\n✅ Triton is ready.")

        print("=" * 60)

    def can_use_triton(self) -> bool:
        """Check if Triton kernels can be used."""
        if self.triton_version is None:
            return False
        if not self.cuda_available:
            return False
        if self.triton_version < (2, 0, 0):
            return False
        return True

    def can_use_flash_attn_triton(self) -> bool:
        """Check if FlashAttention Triton kernel is supported."""
        if not self.can_use_triton():
            return False
        # FlashAttention requires Triton 2.1+ for best performance
        if self.triton_version < (2, 1, 0):
            warnings.warn("Triton < 2.1: FlashAttention may be slower. Consider upgrading.")
        return True

    def get_recommended_backend(self) -> str:
        """Get recommended attention backend."""
        if self.can_use_flash_attn_triton():
            return "triton_flash_attn"

        # Check for flash-attn package
        try:
            import flash_attn
            return "flash_attn_package"
        except ImportError:
            pass

        # Check PyTorch SDPA
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            return "sdpa"

        return "manual"

    def wrap_with_compatibility(self, triton_kernel):
        """Wrap a Triton kernel with version compatibility checks."""
        def wrapper(*args, **kwargs):
            if not self.can_use_triton():
                raise RuntimeError(
                    "Triton not available. "
                    "Install: pip install triton, or use fallback backend."
                )
            return triton_kernel(*args, **kwargs)
        return wrapper


# Global compatibility checker
_compat = None

def get_compatibility() -> TritonCompatibility:
    """Get global compatibility checker."""
    global _compat
    if _compat is None:
        _compat = TritonCompatibility()
    return _compat

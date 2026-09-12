"""Unified multimodal processor for HeliosLM.

Usage:
    processor = MultimodalProcessor(model, vision_encoder, audio_encoder, fusion)

    # Image understanding
    result = processor.process(
        text="Describe this image",
        image="path/to/image.jpg"
    )

    # Audio understanding
    result = processor.process(
        text="Transcribe this audio",
        audio="path/to/audio.wav"
    )

    # Multimodal (image + audio + text)
    result = processor.process(
        text="What is happening in this video?",
        image="frame.jpg",
        audio="audio.wav"
    )
"""
import torch
from typing import Optional, Union


class MultimodalProcessor:
    """Unified interface for multimodal inference."""

    def __init__(self, lm_model, vision_encoder, audio_encoder, fusion_module, tokenizer):
        self.lm = lm_model
        self.vision = vision_encoder
        self.audio = audio_encoder
        self.fusion = fusion_module
        self.tokenizer = tokenizer
        self.device = next(lm_model.parameters()).device

    @torch.no_grad()
    def process(
        self,
        text: str,
        image: Optional[Union[str, torch.Tensor]] = None,
        audio: Optional[Union[str, torch.Tensor]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """
        Process a multimodal query.

        Args:
            text: Text prompt
            image: Image path or tensor
            audio: Audio path or waveform tensor
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            response: Generated text response
        """
        self.lm.eval()

        # Tokenize text
        text_tokens = self.tokenizer.encode(text, add_special_tokens=True)
        text_tensor = torch.tensor([text_tokens], device=self.device)

        # Get text embeddings
        text_embeds = self.lm.embed_tokens(text_tensor)

        # Encode vision if provided
        vision_embeds = None
        if image is not None:
            from vision.image_processor import ImageProcessor
            processor = ImageProcessor()
            if isinstance(image, str):
                image_tensor = processor.preprocess(image).to(self.device)
            else:
                image_tensor = image.to(self.device)

            _, vision_embeds = self.vision(image_tensor)

        # Encode audio if provided
        audio_embeds = None
        if audio is not None:
            from audio.audio_processor import AudioProcessor
            processor = AudioProcessor()
            if isinstance(audio, str):
                audio_tensor = processor.load(audio).unsqueeze(0).to(self.device)
            else:
                audio_tensor = audio.to(self.device)

            audio_embeds = self.audio(audio_tensor)

        # Fuse modalities
        fused = self.fusion(text_embeds, vision_embeds, audio_embeds)

        # Generate with fused features
        # In real implementation, this would feed into LM decoder
        # Simplified: return placeholder
        return f"[Multimodal response for: {text[:50]}...]"

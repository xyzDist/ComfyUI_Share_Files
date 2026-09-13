import torch
import math
import comfy.nested_tensor

class H3BlendLatents:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_original": ("LATENT", {
                    "description": "原始 latent（未去噪）"
                }),
                "latent_denoised": ("LATENT", {
                    "description": "已去噪 latent（例如 denoise=0.5 嘅結果）"
                }),
                "keyframes": ("STRING", {
                    "default": "0:0, 22:0, 44:1",
                    "multiline": False,
                    "placeholder": "例如: 0:0, 22:0, 44:1（幀號為最終影片幀號，值為混合比例 0-1）"
                }),
                "duration": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.1,
                    "max": 3600.0,
                    "step": 0.1,
                    "description": "影片時長（秒）"
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 240.0,
                    "step": 1.0,
                    "description": "影片 FPS"
                }),
                "interpolation": (["linear", "smooth", "step"], {
                    "default": "smooth"
                }),
                "blend_audio": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enable",
                    "label_off": "disable",
                    "description": "啟用：音頻都用同樣比例混合；停用：直接使用已去噪嘅音頻"
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "blend"
    CATEGORY = "xyzdist/H3"

    def parse_keyframes(self, keyframes_str, total_frames, pixel_total_frames, interpolation):
        pairs = []
        for part in keyframes_str.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            frame_str, value_str = part.split(":", 1)
            try:
                pixel_frame = int(frame_str.strip())
                value = float(value_str.strip())
            except ValueError:
                continue
            pairs.append((pixel_frame, value))

        if not pairs:
            print("[H3Blend] ⚠️ 冇有效 keyframe，全部幀設為 0")
            return [0.0] * total_frames

        pairs.sort(key=lambda x: x[0])

        ratio = total_frames / pixel_total_frames
        latent_pairs = []
        for pixel_frame, value in pairs:
            latent_frame = int(round(pixel_frame * ratio))
            latent_frame = max(0, min(latent_frame, total_frames - 1))
            latent_pairs.append((latent_frame, value))

        dedup = {}
        for f, v in latent_pairs:
            dedup[f] = v
        latent_pairs = sorted(dedup.items())

        print(f"[H3Blend] 🔄 換算: {pixel_total_frames} 像素幀 → {total_frames} latent 幀 (ratio={ratio:.4f})")
        print(f"[H3Blend] 🎯 換算後 latent keyframes: {latent_pairs}")

        if latent_pairs[0][0] > 0:
            latent_pairs.insert(0, (0, latent_pairs[0][1]))

        values = []
        for i in range(total_frames):
            prev_frame, prev_val = latent_pairs[0]
            next_frame, next_val = latent_pairs[-1]
            for j in range(len(latent_pairs)):
                if latent_pairs[j][0] <= i:
                    prev_frame, prev_val = latent_pairs[j]
                if latent_pairs[j][0] >= i:
                    next_frame, next_val = latent_pairs[j]
                    break

            if next_frame == prev_frame:
                t = 0.0
            else:
                t = (i - prev_frame) / (next_frame - prev_frame)

            if interpolation == "linear":
                factor = t
            elif interpolation == "smooth":
                factor = 0.5 * (1 - math.cos(math.pi * t))
            elif interpolation == "step":
                factor = 0.0 if t < 0.5 else 1.0
            else:
                factor = t

            values.append(prev_val + (next_val - prev_val) * factor)

        return values

    def blend(self, latent_original, latent_denoised, keyframes, duration, fps, interpolation, blend_audio):
        samples_a = latent_original.get("samples")
        samples_b = latent_denoised.get("samples")

        if not hasattr(samples_a, "is_nested") or not samples_a.is_nested:
            raise ValueError("[H3Blend] latent_original 必須係 NestedTensor")
        if not hasattr(samples_b, "is_nested") or not samples_b.is_nested:
            raise ValueError("[H3Blend] latent_denoised 必須係 NestedTensor")

        video_a, audio_a = samples_a.unbind()
        video_b, audio_b = samples_b.unbind()

        if video_a.ndim == 4:
            video_a = video_a.unsqueeze(0)
            video_b = video_b.unsqueeze(0)
        if audio_a.ndim == 3:
            audio_a = audio_a.unsqueeze(0)
            audio_b = audio_b.unsqueeze(0)

        B, C, F, H, W = video_a.shape
        print(f"[H3Blend] 📐 Video latent: B={B}, C={C}, F={F}, H={H}, W={W}")

        if video_b.shape != video_a.shape:
            raise ValueError(f"[H3Blend] 兩個 video latent 形狀唔同: {tuple(video_a.shape)} vs {tuple(video_b.shape)}")

        # === 計算像素總幀數 ===
        pixel_total_frames = int(round(duration * fps))
        print(f"[H3Blend] 🎬 Duration={duration}s × FPS={fps} = {pixel_total_frames} 像素幀")

        # === 生成每幀混合比例 ===
        factors = self.parse_keyframes(keyframes, F, pixel_total_frames, interpolation)

        # === 混合 video ===
        video_blended = torch.zeros_like(video_a)
        for i, f in enumerate(factors):
            video_blended[:, :, i, :, :] = video_a[:, :, i, :, :] * (1 - f) + video_b[:, :, i, :, :] * f

        # === 混合 audio（可選） ===
        if blend_audio:
            # audio 時間軸長度同 video 唔同，需要按比例映射 factor
            audio_T = audio_a.shape[-1]
            audio_blended = torch.zeros_like(audio_a)
            for t in range(audio_T):
                # 將 audio frame 映射到 video frame
                video_idx = int(round(t / max(1, audio_T - 1) * (F - 1)))
                f = factors[video_idx]
                audio_blended[..., t] = audio_a[..., t] * (1 - f) + audio_b[..., t] * f
            print(f"[H3Blend] 🔊 Audio 已混合（{audio_T} 個 audio 幀）")
        else:
            audio_blended = audio_b
            print(f"[H3Blend] 🔊 Audio 直接使用已去噪版本")

        # === 輸出 ===
        output = dict(latent_original)
        output["samples"] = comfy.nested_tensor.NestedTensor((video_blended, audio_blended))

        print(f"[H3Blend] 📈 Factor: Frame 0 = {factors[0]:.4f}, Frame {F-1} = {factors[-1]:.4f}")
        print(f"[H3Blend] ✅ 完成")

        return (output,)
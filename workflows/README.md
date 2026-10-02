# Workflows

ComfyUI API-format graphs from the case study. Replace `#inputs/FIRST_FRAME.png` (and `#inputs/LAST_FRAME.png` for the tween) with your key drawings; each prompt is an editable example from the study.

| File | What it is | Input | Length |
| --- | --- | --- | --- |
| `anime_i2v_wan30_first_frame.json` | Wan 3.0, first frame | First frame | 5 s at the model's native rate (30 fps) |
| `anime_i2v_minimax_h3_api_first_frame.json` | MiniMax H3 API, first frame | First frame | 5 s at the model's native rate (24 fps) |
| `anime_seq_adapter_first_frame_124f.json` | Seq adapter, long first-frame generation | First frame (one reference) | 124 frames at 24 fps |
| `anime_seq_adapter_tween_first_last_22f.json` | Seq adapter, tween between two keys | First and last frame (two references) | 22 frames at 24 fps |

The Wan 3.0 and MiniMax H3 API workflows use Floyo partner nodes (`AlibabaWan30ImageToVideo_floyo`, `MiniMaxH3FirstLastFrameToVideo_floyo`) and run on Floyo. The two seq adapter workflows use the core ComfyUI MiniMax H3 nodes.

**Setting up the seq adapter.** Both seq workflows load our seq adapter in their LoRA node (LoraLoaderModelOnly). To set it up:

1. Request access at [alvdansen/h3-keyframe-animation](https://huggingface.co/alvdansen/h3-keyframe-animation) on Hugging Face.
2. Take `h3_seq_step12000` from the repository's `adapters` folder. It is already converted for H3.
3. Put it in your ComfyUI `models/loras` folder and select it in that node.
4. The model card gives the license, the inference settings and the conditioning contract. The adapters run on the full H3 model under MiniMax's H3 license. H3 Turbo is a separate model, outside what they were trained for.

**What the long first-frame workflow asks of it.** The seq adapter was trained on two references, a window's first drawing and its natural end, rendered as 22 frames. The long first-frame workflow gives it one reference and 124 frames, which asks more of it; Part A records how it behaves there. The tween workflow matches its training.

**Changing the tween's length.** The tween's prompt opens with an alignment line that puts Picture 2 at the 0.88-second mark, the key-to-key time of 22 frames at 24 fps. If you change CLIP LENGTH, change that mark to (frames - 1) / 24 seconds for the frame count the length snaps to: 39 frames, for example, is the 1.58-second mark.

**H3 open weights.** Open either seq workflow and set the LoRA strength to 0, or bypass the LoRA node. That runs the open weights on the reference-to-video path. The H3 open-weights clips in the case study came from MiniMax's image-to-video graph, so results with the LoRA off can differ from them.

[Floyo workflow link: to come]

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1920
HEIGHT = 540
FRAME_COUNT = 48

DATA = Path(__file__).with_name("scdd_real_trajectory_sample8.json")
OUT = Path(__file__).with_name("scdd_self_correction_without_remasking.gif")

FONT_DIR = Path("/usr/share/fonts/dejavu-sans-fonts")
SANS = FONT_DIR / "DejaVuSans.ttf"
SANS_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"

PAPER = "#ffffff"
INK = "#151922"
TEXT = "#303744"
FRESH = "#05070a"
MUTED = "#7b8491"
MASK = "#d9dee6"
FAINT = "#eef1f5"


def load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
  return ImageFont.truetype(str(path), size)


TOKEN_FONT = load_font(SANS, 12)
TOKEN_BOLD = load_font(SANS_BOLD, 12)
TITLE_FONT = load_font(SANS, 18)
SMALL_FONT = load_font(SANS, 14)


def state_indices(num_states: int) -> list[int]:
  # States are: all-mask initial, 128 denoising steps, final noise removal.
  last = num_states - 1
  middle = []
  for i in range(FRAME_COUNT - 2):
    progress = i / max(FRAME_COUNT - 3, 1)
    middle.append(round((progress ** 1.08) * (last - 1)))
  indices = [0]
  for idx in middle:
    if idx != indices[-1]:
      indices.append(idx)
  if indices[-1] != last - 1:
    indices.append(last - 1)
  indices.append(last)
  return indices


def display_text(text: str) -> str:
  text = text.replace("\n", " ").replace("\t", " ")
  text = "".join(ch for ch in text if ch.isprintable())
  return text.strip()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
  if not text:
    return 0
  bbox = draw.textbbox((0, 0), text, font=font)
  return bbox[2] - bbox[0]


def layout_groups(payload: dict, draw: ImageDraw.ImageDraw) -> list[tuple[int, int] | None]:
  groups = payload["visualization"]["groups"]
  positions: list[tuple[int, int] | None] = []
  left = 62
  right = 1858
  x = left
  y = 96
  line_h = 20
  word_gap = 13
  para_gap = 9

  for group in groups:
    final_text = group["final_text"]
    if "\n" in final_text:
      positions.append(None)
      x = left
      y += para_gap if final_text == "\n" else line_h
      continue

    text = display_text(final_text)
    if not text:
      positions.append(None)
      continue

    width = max(
        text_width(draw, text, TOKEN_FONT),
        text_width(draw, text, TOKEN_BOLD),
        7) + 2
    gap = word_gap if x > left else 0
    if x + gap + width > right:
      x = left
      y += line_h
      gap = 0
    x += gap
    positions.append((x, y))
    x += width
  return positions


def draw_group(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    final_text: str,
    state_group: dict,
    prev_group: dict | None,
) -> None:
  text = display_text(state_group["text"])
  if state_group["masked"]:
    placeholder = "[MASK]" if text_width(draw, display_text(final_text), TOKEN_FONT) > 42 else "·"
    draw.text((x, y), placeholder, font=TOKEN_FONT, fill=MASK)
    return

  if not text:
    return

  changed = prev_group is not None and (
      state_group["text"] != prev_group["text"]
      or state_group["partial"] != prev_group["partial"])
  font = TOKEN_BOLD if changed else TOKEN_FONT
  fill = FRESH if changed else TEXT
  draw.text((x, y), text, font=font, fill=fill)


def make_frame(payload: dict, state_idx: int, prev_idx: int | None) -> Image.Image:
  states = payload["states"]
  mask_index = payload["mask_index"]
  state = states[state_idx]
  visual = payload["visualization"]
  visual_state = visual["states"][state_idx]
  visual_prev = visual["states"][prev_idx] if prev_idx is not None else None

  image = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
  draw = ImageDraw.Draw(image)

  step_label = "final" if state_idx == len(states) - 1 else f"step {state_idx:03d} / 128"
  draw.text((62, 28), "SCDD real generation trajectory", font=TITLE_FONT, fill=INK)
  draw.text((62, 53), "512-token trajectory rendered as decoded text flow", font=SMALL_FONT, fill=MUTED)
  draw.text((1710, 36), step_label, font=SMALL_FONT, fill=MUTED)
  draw.line((62, 78, 1858, 78), fill=FAINT, width=2)

  positions = layout_groups(payload, draw)
  for group_idx, pos in enumerate(positions):
    if pos is None:
      continue
    x, y = pos
    if y > 468:
      continue
    draw_group(
        draw,
        x,
        y,
        visual["groups"][group_idx]["final_text"],
        visual_state[group_idx],
        visual_prev[group_idx] if visual_prev is not None else None)

  mask_count = sum(1 for token_id in state if token_id == mask_index)
  decoded_count = len(state) - mask_count
  draw.line((62, 490, 1858, 490), fill=FAINT, width=2)
  footer = (
      f"decoded {decoded_count:03d}/512    "
      f"remaining masks {mask_count:03d}    "
      "non-mask tokens revise directly"
  )
  draw.text((62, 505), footer, font=SMALL_FONT, fill=MUTED)
  return image


def main() -> None:
  payload = json.loads(DATA.read_text(encoding="utf-8"))
  indices = state_indices(len(payload["states"]))
  frames = []
  durations = []
  prev_idx = None
  for idx in indices:
    frames.append(make_frame(payload, idx, prev_idx))
    if idx == 0:
      durations.append(520)
    elif idx == len(payload["states"]) - 1:
      durations.append(980)
    else:
      durations.append(55)
    prev_idx = idx

  paletted = [
      frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=96)
      for frame in frames
  ]
  paletted[0].save(
      OUT,
      save_all=True,
      append_images=paletted[1:],
      duration=durations,
      loop=0,
      optimize=True,
      disposal=2)
  print(f"Wrote {OUT}")


if __name__ == "__main__":
  main()

"""Render GridWorld episode frames as an animated GIF or PNG image sequence."""

from pathlib import Path

import pygame


CELL_SIZE = 72
BACKGROUND = (245, 247, 250)
GRID_LINE = (210, 216, 224)
GOLD = (212, 162, 28)
GOLD_HIGHLIGHT = (255, 232, 126)
GOLD_SHADOW = (145, 96, 9)
PURPLE = (128, 69, 171)
GOAL_GREEN = (54, 150, 77)
ROBOT_GREY = (132, 143, 153)
ROBOT_DARK = (65, 74, 83)
KEY_COLOURS = {"R": (220, 58, 58), "B": (58, 115, 211), "E": (51, 161, 89), "Y": (242, 207, 47)}


def _cell_rect(row, col):
    return pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)


def _draw_key(surface, rect, colour):
    """Draw a key with a round bow, shaft, and two teeth."""
    centre = (rect.x + 25, rect.y + 28)
    pygame.draw.circle(surface, colour, centre, 13)
    pygame.draw.circle(surface, BACKGROUND, centre, 5)
    pygame.draw.rect(surface, colour, (rect.x + 33, rect.y + 25, 24, 7), border_radius=3)
    pygame.draw.rect(surface, colour, (rect.x + 49, rect.y + 30, 6, 10))
    pygame.draw.rect(surface, colour, (rect.x + 41, rect.y + 30, 6, 7))


def _draw_bell(surface, rect):
    """Draw a shiny gold bell with a dome, lip, and clapper."""
    cx, cy = rect.center
    pygame.draw.circle(surface, GOLD, (cx, cy - 2), 19)
    pygame.draw.rect(surface, GOLD, (cx - 19, cy - 2, 38, 16), border_radius=4)
    pygame.draw.rect(surface, GOLD_SHADOW, (cx - 23, cy + 12, 46, 5), border_radius=2)
    pygame.draw.circle(surface, GOLD_SHADOW, (cx, cy + 21), 5)
    pygame.draw.rect(surface, GOLD, (cx - 4, cy - 27, 8, 7), border_radius=2)
    pygame.draw.ellipse(surface, GOLD_HIGHLIGHT, (cx - 12, cy - 17, 10, 7))


def _draw_agent(surface, rect):
    """Draw a compact grey robot marker for the agent."""
    body = pygame.Rect(rect.x + 17, rect.y + 20, 38, 35)
    pygame.draw.rect(surface, ROBOT_GREY, body, border_radius=9)
    pygame.draw.rect(surface, (225, 241, 255), (rect.x + 23, rect.y + 29, 26, 10), border_radius=4)
    pygame.draw.circle(surface, ROBOT_DARK, (rect.x + 30, rect.y + 34), 3)
    pygame.draw.circle(surface, ROBOT_DARK, (rect.x + 42, rect.y + 34), 3)
    pygame.draw.line(surface, ROBOT_GREY, (rect.centerx, rect.y + 20), (rect.centerx, rect.y + 12), 3)
    pygame.draw.circle(surface, GOLD, (rect.centerx, rect.y + 10), 3)


def _draw_skull(surface, rect):
    """Draw a compact skull emblem inside a hazard triangle."""
    cx, cy = rect.center
    bone = (239, 230, 215)
    pygame.draw.circle(surface, bone, (cx, cy - 3), 12)
    pygame.draw.rect(surface, bone, (cx - 8, cy + 4, 16, 10), border_radius=2)
    pygame.draw.circle(surface, PURPLE, (cx - 5, cy - 4), 3)
    pygame.draw.circle(surface, PURPLE, (cx + 5, cy - 4), 3)
    pygame.draw.polygon(surface, PURPLE, [(cx, cy), (cx - 2, cy + 4), (cx + 2, cy + 4)])
    pygame.draw.line(surface, PURPLE, (cx - 5, cy + 9), (cx + 5, cy + 9), 2)


def _draw_bell_ring(surface, ring, phase):
    """Animate expanding sound waves for one discrete bell-ring event."""
    row, col = ring["position"]
    rect = _cell_rect(row, col)
    radius = 16 + phase * 11
    colour = GOLD_HIGHLIGHT if phase < 2 else GOLD
    pygame.draw.circle(surface, colour, rect.center, radius, 3)
    pygame.draw.circle(surface, colour, rect.center, radius + 8, 2)
    font = pygame.font.Font(None, 22)
    label = font.render(f"RING {ring['ring_number']}", True, GOLD_SHADOW)
    label_rect = label.get_rect(center=(rect.centerx, max(10, rect.y - 8)))
    surface.blit(label, label_rect)


def _draw_cell(surface, symbol, row, col):
    rect = _cell_rect(row, col)
    pygame.draw.rect(surface, BACKGROUND, rect)
    pygame.draw.rect(surface, GRID_LINE, rect, 1)
    if symbol == "#":
        pygame.draw.rect(surface, (25, 28, 33), rect)
    elif symbol in KEY_COLOURS:
        _draw_key(surface, rect, KEY_COLOURS[symbol])
    elif symbol == "C":
        pygame.draw.circle(surface, GOLD, rect.center, 17)
        pygame.draw.circle(surface, GOLD_HIGHLIGHT, (rect.centerx - 5, rect.centery - 5), 5)
        pygame.draw.circle(surface, GOLD_SHADOW, rect.center, 17, 2)
        font = pygame.font.Font(None, 34)
        dollar = font.render("$", True, GOLD_SHADOW)
        surface.blit(dollar, dollar.get_rect(center=rect.center))
    elif symbol == "S":
        _draw_bell(surface, rect)
    elif symbol == "H":
        points = [(rect.centerx, rect.y + 13), (rect.x + 14, rect.bottom - 13), (rect.right - 14, rect.bottom - 13)]
        pygame.draw.polygon(surface, PURPLE, points)
        pygame.draw.polygon(surface, (87, 41, 120), points, 2)
        _draw_skull(surface, rect)
    elif symbol == "G":
        pygame.draw.line(surface, GOAL_GREEN, (rect.x + 24, rect.y + 13), (rect.x + 24, rect.bottom - 12), 5)
        pygame.draw.polygon(surface, GOAL_GREEN, [(rect.x + 26, rect.y + 15), (rect.right - 13, rect.y + 24), (rect.x + 26, rect.y + 34)])
    elif symbol == "A":
        _draw_agent(surface, rect)


def render_frame(frame, bell_ring=None, ring_phase=0):
    """Render one newline-separated grid frame to a pygame surface."""
    rows = [row.split() for row in frame.splitlines() if row.strip()]
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Grid frame must be a non-empty rectangular grid.")
    surface = pygame.Surface((len(rows[0]) * CELL_SIZE, len(rows) * CELL_SIZE))
    for row_index, row in enumerate(rows):
        for col_index, symbol in enumerate(row):
            _draw_cell(surface, symbol, row_index, col_index)
    if bell_ring is not None:
        _draw_bell_ring(surface, bell_ring, ring_phase)
    return surface


def export_grid_animation(
    frames,
    output_path="outputs/grid_episode.gif",
    frame_duration_ms=220,
    bell_rings=None,
):
    """Export episode frames to a GIF; save PNG frames if Pillow is unavailable."""
    if not frames:
        raise ValueError("Cannot export an animation without episode frames.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pygame.init()
    rings_by_frame = {
        ring["frame_index"]: ring
        for ring in (bell_rings or [])
    }
    surfaces = []
    durations = []
    for index, frame in enumerate(frames):
        ring = rings_by_frame.get(index)
        if ring is None:
            surfaces.append(render_frame(frame))
            durations.append(frame_duration_ms)
        else:
            # Three short expanding-wave frames make each interaction visibly
            # distinct, so two pings are unmistakably two separate rings.
            for phase in range(3):
                surfaces.append(render_frame(frame, ring, phase))
                durations.append(90)
    try:
        from PIL import Image
    except ImportError:
        frame_directory = output.with_suffix("")
        frame_directory.mkdir(parents=True, exist_ok=True)
        for index, surface in enumerate(surfaces):
            pygame.image.save(surface, frame_directory / f"frame_{index:04d}.png")
        pygame.quit()
        return frame_directory, False

    images = [
        Image.frombytes("RGB", surface.get_size(), pygame.image.tostring(surface, "RGB"))
        for surface in surfaces
    ]
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    pygame.quit()
    return output, True

"""
Metaloop: Revolution From My Bed
---------------------------------

A tiny wrapper around the existing MetaLoopOrchestrator that
generates a themed HTML artifact reflecting:
- "so i start a revolution from my bed"
- "arab money"
- "sam & claude"

It runs the metaloop, renders the default cathedral HTML, then
lightly rewrites the hero section text to match the theme.
"""

import asyncio
from pathlib import Path

# Import the existing orchestrator from the agents module
from agents.metaloop import MetaLoopOrchestrator


THEME_TITLE = "METALOOP — REVOLUTION FROM MY BED"
THEME_SUBTITLE_PRIMARY = "so i start a revolution from my bed"
THEME_SUBTITLE_SECONDARY = "arab money • sam & claude"


async def main() -> None:
    orchestrator = MetaLoopOrchestrator(num_iterations=2)

    # Run the metaloop (base -> meta -> meta-meta-lite)
    await orchestrator.run_metaloop()

    # Get default cathedral HTML
    html = orchestrator.generate_cathedral_html()

    # Minimal, safe replacements in hero section
    html = html.replace("<h1>METALOOP</h1>", f"<h1>{THEME_TITLE}</h1>")
    html = html.replace(
        '<p class="subtitle">The Ultimate Recursive Abstraction</p>',
        f'<p class="subtitle">{THEME_SUBTITLE_PRIMARY}</p>'
    )
    # Append second tagline beneath the hero canvas line if present
    marker = '</div>\n\n        <div class="quote">'
    if marker in html:
        html = html.replace(
            marker,
            f'\n        <p class="subtitle" style="margin-top: 0.5rem; color: #aaa;">{THEME_SUBTITLE_SECONDARY}</p>\n\n        <div class="quote">'
        )

    out_path = Path("docs/metaloop_revolution.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    print(f"✨ Wrote themed metaloop to {out_path.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())


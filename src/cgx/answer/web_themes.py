"""Curated design-system themes for the Site Studio.

Design quality can't be prompted out of a mid-size local model -- asked to
"make it beautiful" it improvises bland gray-on-white. So the *design* comes
from a curated library of hand-tuned themes (real palette hex + a Google Font
pairing + component/hero treatment), and generation is CONSTRAINED to the
selected theme's tokens. The model stops inventing colors and instead fills in
a professionally-designed system -- which is what makes a weak model produce a
good-looking site.

A theme is selected deterministically from the brief (keyword match), so a
coffee shop reliably gets a warm espresso palette without the model choosing.
``theme_prompt_block`` renders the theme into hard constraints for the
generator. The starter set below is hand-authored; it is refined/extended from
the ``web-design-system`` workflow's vetted output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

__all__ = ["Theme", "THEMES", "select_theme", "theme_prompt_block", "theme_summaries"]


@dataclass
class Theme:
    key: str
    name: str
    vibe: str
    match_keywords: List[str]
    palette: Dict[str, str]  # bg, surface, text, muted, primary, primary_contrast, accent, border
    fonts: Dict[str, str]    # heading, body, google_link
    style: str
    dark: bool = False


def _link(*families: str) -> str:
    # families like "Playfair Display:wght@600;700"
    fam = "&".join("family=" + f.replace(" ", "+") for f in families)
    return f"https://fonts.googleapis.com/css2?{fam}&display=swap"


THEMES: List[Theme] = [
    Theme(
        key='warm-artisan', name='Warm Artisan',
        vibe='Cozy hand-crafted warmth -- espresso, cream, and caramel with an editorial serif voice.',
        match_keywords=['coffee', 'coffee shop', 'cafe', 'espresso', 'bakery', 'artisan', 'artisan food', 'tea house', 'tea', 'roastery', 'patisserie', 'brunch', 'restaurant', 'chocolate', 'handmade', 'deli'],
        palette={
            'bg': '#FBF6EE',
            'surface': '#FFFDF9',
            'text': '#2B1D14',
            'muted': '#6F5B4E',
            'primary': '#8A4B2A',
            'primary_contrast': '#FFFFFF',
            'accent': '#A85A15',
            'border': '#E7DAC8',
        },
        fonts={
            'heading': 'Fraunces', 'body': 'Nunito Sans',
            'google_link': 'https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;0,9..144,700;1,9..144,500&family=Nunito+Sans:wght@400;500;600;700&display=swap',
        },
        style=(
            'Rounded, tactile, and paper-warm. RADIUS: 6px inputs, 14px cards, 999px pills/buttons. SHA'
            'DOWS are soft and espresso-tinted (never black): cards use 0 2px 4px rgba(43,29,20,.05), 0'
            ' 12px 28px -12px rgba(43,29,20,.18); buttons lift on hover to 0 8px 18px -6px rgba(138,75,'
            '42,.45). SPACING on an 8px base (8/16/24/40/64/96); sections pad 96px vertical desktop / 5'
            '6px mobile; max content width 1120px, 24px gutters. HEADINGS in Fraunces 600-700, tight 1.'
            '1 leading, slightly negative tracking for a display feel; BODY Nunito Sans 400/500 at 1.65'
            ' line-height. PRIMARY BUTTON: solid #8A4B2A with #FFFFFF text (6.7:1), pill, 12px/24px pad'
            'ding, darken to #713A1F on hover. SECONDARY/GHOST: transparent, 1.5px #E7DAC8 border, #8A4'
            'B2A text, fills to #FBF6EE on hover. CARDS: #FFFDF9 surface, 1px #E7DAC8 border, 14px radi'
            'us, 24-32px padding, optional thin caramel #A85A15 top-accent bar. Links, small labels, an'
            'd uppercase eyebrows/kickers (letter-spacing .12em) use accent #A85A15 -- it clears AA on '
            'body text (4.7-5.0:1); underline-on-hover with 4px offset. Dividers and input borders use '
            '#E7DAC8. HERO BACKGROUND (offline-safe, palette-only): stack a warm diagonal gradient line'
            'ar-gradient(135deg,#FBF6EE 0%,#F3E7D4 55%,#EBD9BF 100%), a soft top-left glow radial-gradi'
            'ent(120% 90% at 15% 10%, rgba(168,90,21,.16), transparent 60%), and a faint linen texture '
            'repeating-linear-gradient(45deg, rgba(43,29,20,.025) 0 2px, transparent 2px 9px) -- multip'
            'le CSS background layers, zero remote assets. Anchor the hero with a 3px #8A4B2A rule or a'
            ' pill CTA on the cream.'
        ),
        dark=False),
    Theme(
        key='clean-tech', name='Clean Tech',
        vibe='Crisp, modern, trustworthy -- an airy indigo-accented SaaS look on cool neutrals and generous whitespace.',
        match_keywords=['saas', 'startup', 'app', 'software', 'agency', 'tech', 'product', 'platform', 'b2b', 'dashboard', 'landing', 'fintech', 'api', 'analytics', 'cloud', 'developer'],
        palette={
            'bg': '#F7F8FA',
            'surface': '#FFFFFF',
            'text': '#0F172A',
            'muted': '#5A6478',
            'primary': '#4F46E5',
            'primary_contrast': '#FFFFFF',
            'accent': '#7C3AED',
            'border': '#E2E8F0',
        },
        fonts={
            'heading': 'Space Grotesk', 'body': 'Inter',
            'google_link': 'https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap',
        },
        style=(
            'Geometric, airy, and confident. RADIUS: 12px cards/inputs, 10px buttons, 999px pills/badge'
            's. SHADOWS soft and layered: cards 0 1px 2px rgba(15,23,42,.05), 0 8px 24px rgba(15,23,42,'
            '.06); hover lifts to 0 12px 32px rgba(79,70,229,.12). SPACING on an 8px scale (8/16/24/32/'
            '48/64); sections 96px vertical desktop / 56px mobile; max width ~1120px centered, 24px gut'
            'ters. HEADINGS Space Grotesk 600-700, tight -0.02em tracking, large scale (hero clamp ~40-'
            '>56px, h2 ~36px); BODY Inter 400/500 at 16-18px, line-height 1.6, muted copy #5A6478 (5.6:'
            '1+). BUTTONS: primary solid #4F46E5 with #FFFFFF text (6.3:1), 10px radius, 12px/20px padd'
            'ing, 500 weight, 0 2px 8px rgba(79,70,229,.25) shadow deepening on hover with -1px transla'
            'teY; secondary #FFFFFF surface, 1px #E2E8F0 border, #0F172A text; text links #4F46E5. CARD'
            'S: #FFFFFF, 1px #E2E8F0 border, 12px radius, 24-32px padding; feature icons in a 44px roun'
            'ded-square tile filled rgba(79,70,229,.10) with an indigo glyph. SECTIONS alternate #F7F8F'
            'A and #FFFFFF for rhythm; 1px #E2E8F0 hairline dividers; small uppercase eyebrow labels in'
            ' accent #7C3AED (5.4:1+), letter-spacing .08em. HERO BACKGROUND (offline-safe): base linea'
            'r-gradient(160deg,#F7F8FA 0%,#FFFFFF 55%) under two soft glows radial-gradient(60% 70% at '
            '18% 8%, rgba(79,70,229,.16), transparent 60%) and radial-gradient(50% 60% at 88% 0%, rgba('
            '124,58,237,.12), transparent 60%), optionally over a faint blueprint grid via repeating-li'
            'near-gradients rgba(15,23,42,.04) at 40px; headline #0F172A with accent words filled linea'
            'r-gradient(90deg,#4F46E5,#7C3AED) via background-clip:text. No remote images.'
        ),
        dark=False),
    Theme(
        key='bold-dark', name='Neon Noir',
        vibe='Cinematic near-black canvas lit by an electric violet-to-teal glow -- loud, confident, after-dark.',
        match_keywords=['creative studio', 'portfolio', 'agency', 'nightlife', 'gaming', 'music', 'product launch', 'developer', 'tech', 'events', 'dj', 'photography', 'brand', 'film'],
        palette={
            'bg': '#08080C',
            'surface': '#12121B',
            'text': '#EAEAF2',
            'muted': '#9EA0B8',
            'primary': '#B383FF',
            'primary_contrast': '#0A0A12',
            'accent': '#2DE2C9',
            'border': '#26263A',
        },
        fonts={
            'heading': 'Space Grotesk', 'body': 'Inter',
            'google_link': 'https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap',
        },
        style=(
            'RADIUS: 14px cards/inputs, 999px pills/buttons. SHADOW depth via layered dark shadows plus'
            ' a colored glow: cards 0 1px 0 rgba(255,255,255,.04) inset, 0 12px 40px rgba(0,0,0,.55); p'
            'rimary buttons add a halo 0 0 0 1px rgba(179,131,255,.4), 0 8px 30px rgba(179,131,255,.35)'
            '. SPACING on an 8px base (8/16/24/40/64/96); sections 96-120px vertical, 24px gutters, max'
            ' width ~1120px. BUTTONS: primary solid #B383FF with #0A0A12 text (7.1:1), 12px/24px paddin'
            'g, uppercase .04em label tracking, hover lifts glow and shifts to linear-gradient(135deg,#'
            'B383FF,#2DE2C9); secondary transparent, 1px #26263A border, #EAEAF2 text, border brightens'
            ' to #B383FF on hover. CARDS: #12121B surface, 1px #26263A border, 14px radius, subtle top-'
            'edge highlight, hover raises shadow and glows violet. HEADINGS Space Grotesk 600-700, -0.0'
            '2em tracking, clamp hero 3rem->5.5rem; BODY Inter 400/500 at 1.0625rem, line-height 1.65, '
            'muted #9EA0B8 (7.2:1+) for secondary copy. Accent teal #2DE2C9 clears AA on dark (11-12:1)'
            ' so it is safe for links, active nav, kicker labels, and hairline highlights. Dividers use'
            ' #26263A. HERO BACKGROUND (offline-safe, pure CSS): base #08080C with radial-gradient(60% '
            '80% at 20% 15%, rgba(179,131,255,.22), transparent 60%), radial-gradient(55% 70% at 85% 25'
            '%, rgba(45,226,201,.16), transparent 60%), linear-gradient(180deg,#12121B 0%,#08080C 70%);'
            ' overlay a faint dot grid radial-gradient(rgba(255,255,255,.05) 1px, transparent 1px) at b'
            'ackground-size:22px 22px. No remote images.'
        ),
        dark=True),
    Theme(
        key='elegant-luxury', name='Elegant Luxury',
        vibe="Hushed, deep-toned refinement with warm gold accents -- the quiet confidence of a fine-dining room or a jeweler's velvet tray.",
        match_keywords=['fine dining', 'restaurant', 'hotel', 'resort', 'jewelry', 'jeweler', 'fashion', 'boutique', 'spa', 'luxury', 'couture', 'atelier', 'champagne', 'concierge', 'wine', 'winery', 'gallery'],
        palette={
            'bg': '#14110E',
            'surface': '#1F1A16',
            'text': '#F2EBE0',
            'muted': '#A89A88',
            'primary': '#C9A24B',
            'primary_contrast': '#14110E',
            'accent': '#E4CE9B',
            'border': '#342C24',
        },
        fonts={
            'heading': 'Cormorant Garamond', 'body': 'Jost',
            'google_link': 'https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Jost:wght@300;400;500;600&display=swap',
        },
        style=(
            'RADIUS restrained: 2px buttons/inputs, 6px cards, 0 on full-bleed sections; nothing pill-s'
            'haped (roundedness reads casual, not couture). SHADOWS near-invisible and warm: cards 0 1p'
            'x 2px rgba(0,0,0,.4), 0 12px 40px -12px rgba(0,0,0,.55); emphasis comes from the 1px #342C'
            '24 hairline plus inner top highlight inset 0 1px 0 rgba(228,206,155,.06). SPACING generous'
            ' on an 8px base with large section padding clamp(5rem,10vw,9rem) vertical, max width ~1120'
            'px. HEADINGS Cormorant Garamond 500/600, tight -0.01em tracking, big scale against the 300'
            '-weight Jost body (letter-spacing .01em, 1.7 line-height). Uppercase Jost 500 with .22em l'
            'etter-spacing for eyebrows/kickers and nav -- the signature luxury small-caps cue. BUTTONS'
            ': primary solid gold #C9A24B with #14110E text (7.8:1), no shadow, hover to champagne #E4C'
            'E9B on a slow 240ms ease; secondary transparent with 1px gold border and gold text that fi'
            'lls on hover. Gold #C9A24B and champagne #E4CE9B both clear AA on the dark ground (7-12:1)'
            ' so they are safe for links, prices, and label text. CARDS: #1F1A16 surface, hairline bord'
            'er, a thin 1px gold top-rule or a small gold serif numeral, 2rem+ padding. Dividers are sh'
            'ort 60px centered #342C24 rules with a gold midpoint dot; links underline on hover with a '
            '1px gold underline offset 4px. HERO BACKGROUND (offline-safe, palette-only): radial-gradie'
            'nt(120% 90% at 70% 15%, rgba(201,162,75,.16) 0%, rgba(201,162,75,0) 42%), linear-gradient('
            '180deg,#1A1611 0%,#14110E 55%,#100D0B 100%); optionally a whisper-thin gold hairline grid '
            'repeating-linear-gradient(90deg, rgba(228,206,155,.035) 0 1px, transparent 1px 88px). No r'
            'emote images.'
        ),
        dark=True),
    Theme(
        key='fresh-natural', name='Fresh & Natural',
        vibe='Calm, organic wellness -- soft sage greens and warm cream with airy, rounded, breathable layouts.',
        match_keywords=['wellness', 'salon', 'spa', 'plants', 'eco', 'clinic', 'yoga', 'organic', 'health', 'beauty', 'nutrition', 'garden', 'sustainability', 'meditation', 'skincare', 'botanical', 'fitness', 'therapy'],
        palette={
            'bg': '#F7F4ED',
            'surface': '#FFFFFF',
            'text': '#22302A',
            'muted': '#5F6B60',
            'primary': '#3F7150',
            'primary_contrast': '#FFFFFF',
            'accent': '#D98E5A',
            'border': '#E4DECE',
        },
        fonts={
            'heading': 'Poppins', 'body': 'Nunito Sans',
            'google_link': 'https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Nunito+Sans:wght@400;600;700&display=swap',
        },
        style=(
            'Rounded, soft, and airy. RADIUS: cards/images 20px, buttons/inputs full-pill (999px), smal'
            'l chips 12px. SHADOWS gentle and green-tinted: cards 0 6px 20px rgba(34,48,42,.06) with a '
            '1px #E4DECE border; hover lifts to 0 12px 28px rgba(34,48,42,.10) with -2px translateY. SP'
            'ACING on an 8px scale (8/16/24/40/64/96); sections 96px vertical, max width ~1120px so eve'
            'rything breathes. HEADINGS Poppins 600/700, -0.01em tracking, 1.15 line-height; BODY Nunit'
            'o Sans 400 at 1.7 line-height; muted copy #5F6B60 (5.1:1+). PRIMARY BUTTON: solid #3F7150 '
            'with #FFFFFF text (5.7:1), pill, 14px/28px padding, subtle 0 4px 12px rgba(63,113,80,.25),'
            ' hover darkens ~8% with a soft glow. SECONDARY: transparent, 1.5px #3F7150 outline, green '
            'text. IMPORTANT accent rule: terracotta #D98E5A is DECORATIVE ONLY -- icon glyphs, tag/chi'
            'p fills (always with dark #22302A ink over them), underline dots, and thin top-accent bars'
            '; it does NOT meet text contrast on light backgrounds (2.4-2.6:1) so it is NEVER used as b'
            'ody text, link text, or a headline color. For colored text/link emphasis use primary #3F71'
            '50 (5.2-5.7:1). CARDS on #FFFFFF above the #F7F4ED page, often with a small circular sage '
            'icon badge (rgba(63,113,80,.12) fill). Section dividers use thin #E4DECE hairlines and sof'
            't organic shapes. HERO BACKGROUND (offline-safe): radial-gradient(1200px 600px at 15% -10%'
            ', rgba(63,113,80,.16), transparent 60%), radial-gradient(900px 500px at 95% 0%, rgba(217,1'
            '42,90,.12), transparent 55%), linear-gradient(160deg,#F7F4ED 0%,#EEF1E7 45%,#E7EFE4 100%);'
            ' optional faint botanical dot texture repeating-radial-gradient(circle at center, rgba(63,'
            '113,80,.05) 0 1.5px, transparent 1.5px 22px). No remote images.'
        ),
        dark=False),
    Theme(
        key='playful-vibrant', name='Playful Vibrant',
        vibe='Bright, bouncy and joyful -- candy-colored energy with chunky rounded type and confetti-soft shapes.',
        match_keywords=['kids', 'children', 'events', 'party', 'food truck', 'festival', 'creative', 'playful', 'fun', 'ice cream', 'toys', 'birthday', 'carnival', 'brand', 'retail', 'shop', 'store'],
        palette={
            'bg': '#FFFDF7',
            'surface': '#FFFFFF',
            'text': '#2B1B4A',
            'muted': '#6B6480',
            'primary': '#FF4D8D',
            'primary_contrast': '#2B1B4A',
            'accent': '#00BFA6',
            'border': '#F0E6D8',
        },
        fonts={
            'heading': 'Baloo 2', 'body': 'Nunito',
            'google_link': 'https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;600;700;800&family=Nunito:wght@400;500;600;700&display=swap',
        },
        style=(
            'Rounded and chunky throughout. RADIUS: buttons/inputs 999px pill, cards 24px, media 20px, '
            'chips 12px. SPACING on an 8px base (8/16/24/40/64/96); sections 96px vertical desktop / 56'
            'px mobile, max width 1120px. SHADOWS soft, colored, lifted: cards 0 10px 30px rgba(43,27,7'
            '4,.08) rising to 0 16px 40px rgba(255,77,141,.18) with -3px translateY on hover. BUTTONS: '
            'primary solid #FF4D8D with #2B1B4A ink (4.9:1), 800-weight Baloo 2, 14px/28px padding, pil'
            'l, plus a chunky 0 4px 0 #E23A78 bottom press-shadow collapsing to 0 1px 0 on :active (arc'
            'ade feel); secondary surface-white with a 2px #2B1B4A border and the same press-shadow in '
            'ink. HEADINGS Baloo 2 700-800, -0.01em tracking, 1.1 line-height; BODY Nunito 400-500 at 1'
            '7px/1.7; muted copy #6B6480 (5.5:1+); default body text uses #2B1B4A. COLOR RULES: pink #F'
            'F4D8D is a FILL / large-display color (buttons, hero, oversized headline words) -- do not '
            'set it as small body/link text (only ~3:1 on white). Teal #00BFA6 and a sunny amber #FFC93'
            'C are DECORATIVE fills only -- 6px card top-bars, pill tag backgrounds (always with #2B1B4'
            'A ink over them), confetti dots, blob shapes -- never as text/link color. For text links u'
            'se #2B1B4A with a pink underline. CARDS: #FFFFFF, 2px #F0E6D8 border, 28px padding, thin 6'
            'px accent top-bar rotating #FF4D8D / #00BFA6 / #FFC93C. Section separators can use a dot p'
            'attern radial-gradient(circle,#F0E6D8 2px, transparent 2px) at 24px 24px. HERO (offline-sa'
            'fe, palette-only): base linear-gradient(135deg,#FF4D8D 0%,#FF8A5C 45%,#FFC93C 100%), two d'
            'epth glows radial-gradient(circle at 18% 22%, rgba(0,191,166,.45), transparent 42%) and ra'
            'dial-gradient(circle at 85% 78%, rgba(124,77,255,.40), transparent 45%), plus confetti rad'
            'ial-gradient(circle at 50% 50%, rgba(255,255,255,.35) 2px, transparent 2px) at 28px 28px; '
            'hero text renders in #FFFDF7 with a 0 2px 0 rgba(43,27,74,.25) shadow for pop. All CSS gra'
            'dients/patterns -- no remote images.'
        ),
        dark=False),
]


_DEFAULT_KEY = 'clean-tech'


def _by_key(key: str) -> Optional[Theme]:
    for t in THEMES:
        if t.key == key:
            return t
    return None


def select_theme(brief: str, explicit_key: Optional[str] = None) -> Theme:
    """Pick the best-fitting theme for ``brief`` (deterministic keyword score)."""
    if explicit_key:
        t = _by_key(explicit_key)
        if t:
            return t
    low = (brief or "").lower()
    best: Optional[Theme] = None
    best_score = 0
    for t in THEMES:
        score = sum(1 for kw in t.match_keywords if kw in low)
        if score > best_score:
            best, best_score = t, score
    return best or _by_key(_DEFAULT_KEY) or THEMES[0]


def theme_prompt_block(theme: Theme) -> str:
    """Render a theme into hard design constraints for the generator."""
    p = theme.palette
    gl = theme.fonts["google_link"]
    heading = theme.fonts["heading"]
    body = theme.fonts["body"]
    scheme = "DARK theme (dark backgrounds, light text)" if theme.dark else "LIGHT theme"
    return (
        f"DESIGN SYSTEM -- you MUST use this exact theme ({theme.name}: {theme.vibe}). "
        f"This is a {scheme}.\n"
        f"- Load these fonts in <head>: "
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        f'<link href="{gl}" rel="stylesheet">\n'
        f"- Headings font: '{heading}'; body font: '{body}'.\n"
        f"- Define these as CSS variables on :root and use them EVERYWHERE "
        f"(never invent other colors):\n"
        f"    --bg:{p['bg']}; --surface:{p['surface']}; --text:{p['text']}; "
        f"--muted:{p['muted']}; --primary:{p['primary']}; "
        f"--primary-contrast:{p['primary_contrast']}; --accent:{p['accent']}; "
        f"--border:{p['border']};\n"
        f"- Page background --bg, body text --text, cards --surface with "
        f"--border; primary buttons use --primary with --primary-contrast text; "
        f"links/highlights use --accent. Ensure strong contrast on every "
        f"section (this palette is designed for it -- do not wash it out).\n"
        f"- STYLE: {theme.style}\n"
        f"(If using Tailwind, wire these into tailwind.config via the CDN "
        f"`tailwind.config = {{...}}` script, or use arbitrary values like "
        f"bg-[var(--primary)]; either way the palette above is authoritative.)"
    )


def theme_summaries() -> List[Dict[str, str]]:
    """Lightweight listing for a UI theme picker."""
    return [{"key": t.key, "name": t.name, "vibe": t.vibe,
             "primary": t.palette["primary"], "bg": t.palette["bg"],
             "accent": t.palette["accent"], "dark": str(t.dark).lower()}
            for t in THEMES]

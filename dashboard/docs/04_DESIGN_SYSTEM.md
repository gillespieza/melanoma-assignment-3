# 04 · Design System

**Bar:** premium clinical-grade software – think Epic / modern medical SaaS. Calm,
trustworthy, dense-but-legible, tabular numerics. Clinical white (chosen direction),
teal/blue accents. NOT dark, NOT playful.

## Tokens (already in `tailwind.config.js`)

```
clinical.bg      #f6f8fb   page background
clinical.panel   #ffffff   cards
clinical.border  #e6ebf1   hairlines
clinical.ink     #0f2033   primary text
clinical.muted   #5b6b7c   secondary text
clinical.teal    #0ea5a4   / tealdark #0f766e   primary accent (immuno)
clinical.blue    #2563eb   / bluedark #1e40af   secondary accent (targeted)
clinical.amber   #d97706   warnings / combination arm
clinical.rose    #e11d48   not-recommended / risk
clinical.green   #16a34a   confirmed / positive
```

Font: **Inter** (loaded in `index.css`). Numerics use `.tabular` (tabular-nums).
Shadows: `shadow-card` (resting), `shadow-lift` (raised/CTA). Radius: cards `rounded-2xl`,
controls `rounded-lg/xl`.

## Conventions

- Wrap every section in `<Panel title icon subtitle right>` from `components/ui.tsx`.
- Status chips → `<Pill tone>`; big numbers → `<Stat>`.
- Method color coding, keep consistent everywhere:
  - **Immunotherapy** = teal (`tealdark`)
  - **Targeted (BRAF/MEK)** = blue
  - **Combination / sequencing** = amber (dashed lines in charts)
  - **Baseline / no therapy** = slate/grey dashed
- Motion: Framer Motion for the simulation overlay + result reveal + tree-node stagger.
  Keep it subtle (0.3–0.4s, easeOut). No bouncing.
- Never use emojis. Avoid heavy borders; prefer whitespace + hairlines.
- Everything must be legible when projected in a lecture theatre: generous font sizes on
  headline numbers, high contrast, no thin grey-on-white body text below ~12px.

## Layout

- Max width ~1400px, centered, `px-6`.
- Sticky `Header` with the `ViewToggle` (Cohort ↔ Patient).
- Cohort view: full-width table + featured archetype cards on top.
- Patient view: left intake/passport rail (sticky) + right lane stack (current v1 layout,
  extended with the new lanes).

## The "agreement" visual (signature moment)

The methods-agreement badge is the centerpiece of Q5. Design it to read instantly:
- **Concordant** → green badge, both method icons lit, "ML + ODE agree".
- **Discordant** → amber badge, "Methods split – consultant review", show both positions.
Put a small two-dot or converging-arrows glyph. This is the thing that makes the multi-method
story legible in one glance – invest in it.

## Quality checklist (the "$100M" bar)

- Consistent 4/8px spacing rhythm; nothing visually misaligned.
- Numbers formatted (1 decimal for %, integer months); units in muted small caps.
- Empty/loading/awaiting states are designed, not raw text.
- Charts: no default recharts styling leaking through – custom axis ticks, hairline grids,
  rounded tooltips (see existing chart components for the established style).
- Hover/active states on every interactive element.

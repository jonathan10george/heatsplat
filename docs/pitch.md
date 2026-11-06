# Heatsplat: the 3-minute pitch

One line: **Walk past a building with a phone and a $250 thermal clip-on; get a 3D map of where its
heat leaves, ranked, with a dollar figure on each leak.**

Tagline on the title slide: *Thermal imaging, but in 3D, and it tells you what it costs.*

## What the research says wins (and how this pitch uses it)

- Judges score **the product, the pitch and the potential, not the code** ([Reskilll's rubric
  breakdown](https://blogs.reskilll.com/what-hackathon-judges-look-for-complete-judging-criteria-breakdown-2026/)).
  So the demo leads, the architecture gets 40 seconds, and the last word is the business.
- **Clarity beats complexity**; write the demo script first and frame the problem around one person
  ([HackerEarth](https://www.hackerearth.com/blog/10-tips-win-hackathon),
  [JetBrains judging notes](https://blog.jetbrains.com/ai/2026/06/how-to-win-a-hackathon-notes-from-the-judging-table/)).
  Our person is the facilities manager who knows a dorm is inefficient "on paper" and cannot say where.
- **One voice per segment, a separate demo driver, and a backup video**: a team lost nothing to a
  dead API because they had the recording ([Music Hackspace's 3-minute pitch guide](https://musichackspace.org/blog/hackathon-presentation-tips)).
  The tour button exists for exactly this: record it once, keep the file on the laptop.
- **The Cube is a venture crowd.** Its site talks about alumni companies and YC/a16z backing; the
  November 2025 Buildathon winner on record is a 3D-aware virtual staging MVP that won on product
  plus pitch (Joe Phelps, [LinkedIn](https://www.linkedin.com/in/joe-phelps/)). Close on the
  customer and the money, not the math.
- Recent grand-prize winners are real-world sensing with a working demo: Hawkwatch (live camera
  alerts, [TreeHacks 2025](https://stanforddaily.com/2025/02/18/treehacks-awards-200000-in-prizes-to-students-from-around-the-world/))
  and Shepherd, a computer-vision cane ([TreeHacks 2026](https://stanforddaily.com/2026/02/15/12th-annual-treehacks/)).
  We are in that lane: a real building, real thermal frames, a live 3D result.

## Four equal parts (45 seconds each)

Rotate the order of speakers between practice runs so nobody owns "the boring part". For a team
of three, merge parts 2 and 4 into one speaker; for two, A takes 1+2 and B takes 3+4.

### Part 1, the hook (Speaker A)

Slide: a thermal photo of a dorm, one orange smear.

"Facilities at Duke knows Gilbert-Addoms is inefficient. A Bass Connections team wrote the paper.
What nobody has is *where*. A thermal photo is a flat orange smear; an energy audit is a week and a
contractor. We walked past a building for fifteen minutes with a phone and a clip-on thermal camera,
and here is where its heat leaves, in 3D, ranked, in dollars."

Cue Speaker C to switch to the app on screen. A says the last sentence while the viewer appears.

### Part 2, how it works (Speaker B)

Slide: the four-box pipeline (phone video, Gaussian splat, thermal bake, detector).

"Three steps. One: the phone video becomes a Gaussian splat, the same tech behind the 3D scans you
have seen on your feed, trained on a laptop GPU in three minutes. Two: every thermal frame is
projected back onto the splat, so each of the ten thousand Gaussians carries a temperature. Three: a
detector looks for patches warmer than the two metres around them, clusters them, and prices each
one with a heat-loss model. The scale is metric, so the areas and the costs are real units."

Keep to three sentences per step. Do not mention the libtorch patch or the calibration bug unless
asked; they are Q&A gold, not pitch material.

### Part 3, the live demo (Speaker C drives, narrates)

Press **space**. The tour runs itself; C talks over it, and the caption bar carries the numbers.

1. Photographic splat orbiting: "This is the building as the phone saw it."
2. Scan line, thermal mode: "Same geometry, now every surface carries its temperature."
3. Leaks mode: "Everything normal goes grey. What is left is warmer than its surroundings."
4. Fly to leak 1, then 2: read the caption: cause, degrees, square metres, dollars a year.
5. Back to the home view with the total: "That is the bill."

Then one manual move: drag the threshold slider right. "Raise the bar and only the worst survive."
Then press **1**, **2**, **3** once each to show the modes respond instantly.

Fallback: if the viewer stalls, play `docs/tour.mp4` (record it from the tour before the event) and
narrate the same five beats.

### Part 4, the numbers and the ask (Speaker D)

Slide: two tiles, "$ per year" and "CO₂ per year", and the camera's price.

"On the synthetic house with planted leaks the detector finds all five. On the real building it
lights the glazed lobby and the window frames, 2 to 3 degrees warmer than the wall, which is exactly
where a glass façade loses heat. Hardware cost: a $250 clip-on and the phone you own. Customer:
campus facilities and the 5 million US commercial buildings that have never had an audit. Business:
a scan per façade, priced against the audit it replaces. Ask: one evening on a Duke dorm with
Facilities' thermal camera, and we hand them the first map."

## Q&A: the honest answers, assigned

| Question | Owner | Answer |
|---|---|---|
| Is the real-building leak a real defect? | D | No: it is the glazing. Glass and frames run 2 to 3 °C warmer than cladding, which is real heat loss, and we say so in the UI. A brick dorm with punched windows is the target. |
| How accurate is the temperature? | B | Within the camera's own accuracy (±3 °C for a FLIR One); the detector works on differences from the surroundings, which cancel most of the offset. |
| Why does the real scan look soft? | B | 96 low-res frames from a public dataset; floaters between camera and wall. Our own capture will have five times the frames at 4K. |
| Where do the dollar figures come from? | D | Watts from the temperature difference, a surface coefficient and the area; hours of heating season; the local tariff. All three are visible and adjustable. |
| What is the scale based on? | C | Storey pitch and the walking stride on this dataset (0.9 m per unit, ±25%); on our own scan we measure a door. |
| Why Gaussian splats and not a mesh? | B | Trains in minutes on a laptop, renders in a browser, and keeps soft edges where thermal data is soft. |
| What breaks it? | A | Rain-wet brick, sun-warmed walls, and glass reflections. We scan 1 to 2 hours after dusk on a dry night. |

## Run of show

- Laptop on power, external display tested, browser at 125% zoom, both servers up
  (`scripts/dev.sh` or the two launch entries), scene preselected to the real building.
- Tour rehearsed with the narration four times; total under 2:50 on a timer.
- `docs/tour.mp4` on the desktop, VLC open behind the browser.
- Slides: title, thermal smear, pipeline, numbers. Four slides, no bullets longer than six words.
- Keys: **space** tour, **1 2 3** modes, **J K** next/previous leak, **H** home, **Esc** stop.

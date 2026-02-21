---
title : "Quantum Fourier Transform"
date : 2026-02-28
mathjax : true
thumbnail: gallery/thumbnails/career_conference_notes.jpeg
cover: gallery/thumbnails/career_conference_notes.jpeg
cover_size: cover
cover_height: 600px
tags:
    - career
categories:
    - opinion
excerpt: "Thoughts "
---


![xxx](quantum-fourier-transform/bloch_frames/1.png)

# A Useful Visualization


{% raw %}

<script>
function toBinary(n) {
  return Number(n).toString(2).padStart(4, '0');
}

function updateBlochFrame(value) {
  document.getElementById("blochImage").src =
    "quantum-fourier-transform/" + value + ".png";

  document.getElementById("frameLabel").innerHTML =
    "Binary: " + toBinary(value);
}
</script>

{% endraw %}


{% raw %}
<div align="center" style="margin-top:40px;">

  <input
    type="range"
    min="0"
    max="7"
    step="1"
    value="0"
    id="blochSlider"
    style="width:60%;"
    oninput="updateBlochFrame(this.value)"
  >

  <p id="frameLabel" style="margin-top:15px;">
    Binary: 0000
  </p>

  <img
    id="blochImage"
    src="quantum-fourier-transform/0.png"
    width="450"
    style="margin-top:20px;"
  >

</div>

{% endraw %}
// Dataset pages: gate filter + text search. Episode pages: arrow keys move to the previous/next episode.
(() => {
  const toolbar = document.querySelector(".toolbar");
  if (toolbar) {
    const buttons = [...toolbar.querySelectorAll("[data-filter]")];
    const search = toolbar.querySelector("input");
    const count = toolbar.querySelector(".count");
    const cards = [...document.querySelectorAll(".card")];
    const groups = [...document.querySelectorAll(".group")];
    const empty = document.querySelector(".empty");
    let gate = "all";
    const apply = () => {
      const terms = search.value.toLowerCase().split(/\s+/).filter(Boolean);
      let shown = 0;
      for (const card of cards) {
        const ok = (gate === "all" || card.dataset.gate === gate) && terms.every((t) => card.dataset.text.includes(t));
        card.hidden = !ok;
        shown += ok;
      }
      for (const group of groups) group.hidden = !group.querySelector(".card:not([hidden])");
      empty.hidden = shown > 0;
      count.textContent = shown === cards.length ? "" : `${shown} of ${cards.length} shown`;
    };
    for (const button of buttons) {
      button.addEventListener("click", () => {
        gate = button.dataset.filter;
        for (const b of buttons) b.setAttribute("aria-pressed", String(b === button));
        apply();
      });
    }
    search.addEventListener("input", apply);
  }
  // Clicking a card's video opens the in-page player with the action generation prompt.
  const player = document.querySelector("dialog.player");
  if (player) {
    const video = player.querySelector("video");
    let current = null;
    const show = (card) => {
      current = card;
      const d = card.dataset;
      video.poster = d.poster;
      video.src = d.clip;
      player.querySelector(".player__id").textContent = d.id;
      player.querySelector("#player-title").textContent = d.title;
      const prompt = player.querySelector(".player__prompt");
      prompt.textContent = d.prompt || d.promptMissing || "No reviewed action prompt yet.";
      prompt.classList.toggle("prompt--missing", !d.prompt);
      player.querySelector(".player__pub").hidden = !d.publisher;
      player.querySelector(".player__publisher").textContent = d.publisher;
      player.querySelector(".player__open").href = d.href;
      history.replaceState(null, "", "#play=" + encodeURIComponent(d.id));
      if (!player.open) player.showModal();
      video.play().catch(() => {});
    };
    const step = (delta) => {
      const visible = [...document.querySelectorAll(".card:not([hidden])")];
      const next = visible[visible.indexOf(current) + delta];
      if (next) show(next);
    };
    for (const button of document.querySelectorAll(".card__img")) {
      button.addEventListener("click", () => show(button.closest(".card")));
    }
    player.querySelector(".player__close").addEventListener("click", () => player.close());
    player.addEventListener("click", (event) => { if (event.target === player) player.close(); });
    player.addEventListener("close", () => {
      video.pause(); video.removeAttribute("src"); video.load();
      if (location.hash.startsWith("#play=")) history.replaceState(null, "", location.pathname + location.search);
    });
    const fromHash = () => {
      const id = decodeURIComponent(location.hash.slice(6));
      const card = location.hash.startsWith("#play=") && [...document.querySelectorAll(".card")].find((c) => c.dataset.id === id);
      if (card) show(card);
    };
    window.addEventListener("hashchange", fromHash);
    fromHash();
    player.addEventListener("keydown", (event) => {
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        step(event.key === "ArrowLeft" ? -1 : 1);
      }
    });
  }
  document.addEventListener("keydown", (event) => {
    if (player && player.open) return;
    if (event.altKey || event.ctrlKey || event.metaKey || event.target.closest("input, video, textarea")) return;
    const link = document.querySelector(event.key === "ArrowLeft" ? '.pager a[rel="prev"]' : event.key === "ArrowRight" ? '.pager a[rel="next"]' : null);
    if (link) location.href = link.href;
  });
})();

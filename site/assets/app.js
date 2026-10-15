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
  document.addEventListener("keydown", (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.target.closest("input, video, textarea")) return;
    const link = document.querySelector(event.key === "ArrowLeft" ? '.pager a[rel="prev"]' : event.key === "ArrowRight" ? '.pager a[rel="next"]' : null);
    if (link) location.href = link.href;
  });
})();

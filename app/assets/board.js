// Drag-reorder for board cards. Panel renders everything inside nested shadow
// roots, so the flex container is found by walking them; Sortable then works
// on the real children, and the resulting order is reported to the server
// through a hidden text input.
(function () {
  function* walk(root) {
    for (const el of root.querySelectorAll("*")) {
      if (el.shadowRoot) yield* walk(el.shadowRoot);
      yield el;
    }
  }

  function findSinkInput() {
    for (const el of walk(document)) {
      if (el.classList && el.classList.contains("order-sink") && el.shadowRoot) {
        const input = el.shadowRoot.querySelector("input");
        if (input) return input;
      }
    }
    return null;
  }

  function report(container) {
    const order = [...container.children]
      .map((c) => (String(c.className).match(/card-(\d+)/) || [])[1])
      .filter(Boolean)
      .join(",");
    const input = findSinkInput();
    if (input && order) {
      input.value = order;
      input.dispatchEvent(new Event("change", { bubbles: true }));
    }
  }

  function attach() {
    if (!window.Sortable) return;
    for (const host of walk(document)) {
      if (!host.classList || !host.classList.contains("board-flex")) continue;
      if (!host.shadowRoot) continue;
      const inner = [...host.shadowRoot.querySelectorAll("*")].find(
        (e) => e.children.length && [...e.children].some((c) => c.shadowRoot)
      );
      if (!inner || inner._npiSortable) continue;
      inner._npiSortable = new Sortable(inner, {
        animation: 150,
        delay: 120,
        delayOnTouchOnly: false,
        ghostClass: "npi-ghost",
        onEnd: () => report(inner),
      });
    }
  }

  setInterval(attach, 800);
})();

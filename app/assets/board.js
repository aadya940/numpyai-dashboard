// Client-side behaviours Panel widgets cannot express: drag-reorder for board
// cards and @-mention autocomplete in the chat input. Panel renders inside
// nested shadow roots, so both find their targets by walking them; state flows
// to and from the server through hidden text inputs.
(function () {
  function* walk(root) {
    for (const el of root.querySelectorAll("*")) {
      if (el.shadowRoot) yield* walk(el.shadowRoot);
      yield el;
    }
  }

  function findSink(cls) {
    for (const el of walk(document)) {
      if (el.classList && el.classList.contains(cls) && el.shadowRoot) {
        const input = el.shadowRoot.querySelector("input");
        if (input) return input;
      }
    }
    return null;
  }

  // -- drag reorder ---------------------------------------------------------

  function reportOrder(container) {
    const order = [...container.children]
      .map((c) => (String(c.className).match(/card-(\d+)/) || [])[1])
      .filter(Boolean)
      .join(",");
    const input = findSink("order-sink");
    if (input && order) {
      input.value = order;
      input.dispatchEvent(new Event("change", { bubbles: true }));
    }
  }

  function attachSortable() {
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
        // Never contest gestures over charts, tables or the control cluster:
        // a click on a bar must reach ECharts, not start a drag. Shadow
        // retargeting hides the real target, so inspect the composed path.
        filter: (evt) => {
          const path = evt.composedPath ? evt.composedPath() : [];
          return path.some(
            (el) =>
              el.tagName === "CANVAS" ||
              (el.classList &&
                (el.classList.contains("card-controls") ||
                  el.classList.contains("tabulator")))
          );
        },
        preventOnFilter: false,
        onEnd: () => reportOrder(inner),
      });
    }
  }

  // -- @mention autocomplete -------------------------------------------------

  function fileNames() {
    const input = findSink("files-sink");
    return input && input.value ? input.value.split(",").filter(Boolean) : [];
  }

  function buildMenu() {
    // Fixed-position and mounted on <body>: the chat input lives inside an
    // overflow:auto container that clips anything absolutely positioned
    // above it - the menu rendered, and no pixel of it was ever visible.
    const menu = document.createElement("div");
    menu.style.cssText =
      "position:fixed;z-index:10000;display:none;" +
      "background:#141d33;border:1px solid rgba(255,255,255,.12);border-radius:10px;" +
      "box-shadow:0 10px 30px rgba(0,0,0,.5);padding:4px;" +
      "min-width:190px;font-size:12.5px;font-family:Inter,system-ui,sans-serif;";
    document.body.appendChild(menu);
    return menu;
  }

  function anchor(menu, textarea) {
    const r = textarea.getBoundingClientRect();
    menu.style.left = Math.round(r.left) + "px";
    menu.style.bottom = Math.round(window.innerHeight - r.top + 6) + "px";
  }

  function attachMention() {
    for (const host of walk(document)) {
      if (!host.classList) continue;
      if (!String(host.className).includes("chatarea")) continue;
      if (!host.shadowRoot || host._npiMention) continue;
      const textarea = host.shadowRoot.querySelector("textarea");
      if (!textarea) continue;
      host._npiMention = true;

      const menu = buildMenu();
      let items = [];
      let active = 0;

      const hide = () => {
        menu.style.display = "none";
        items = [];
      };

      const render = () => {
        menu.innerHTML = "";
        items.forEach((name, i) => {
          const row = document.createElement("div");
          row.textContent = "@" + name;
          row.style.cssText =
            "padding:6px 10px;border-radius:7px;cursor:pointer;" +
            (i === active
              ? "background:#243356;color:#aab8ff;font-weight:600;"
              : "color:#c3cddf;");
          row.addEventListener("mousedown", (e) => {
            e.preventDefault();
            pick(i);
          });
          menu.appendChild(row);
        });
        anchor(menu, textarea);
        menu.style.display = items.length ? "block" : "none";
      };
      window.addEventListener("scroll", hide, true);

      const pick = (i) => {
        const name = items[i];
        const caret = textarea.selectionStart;
        const before = textarea.value.slice(0, caret);
        const after = textarea.value.slice(caret);
        textarea.value =
          before.replace(/@([\w.\-]*)$/, "@" + name + " ") + after;
        textarea.dispatchEvent(new Event("input", { bubbles: true }));
        textarea.focus();
        hide();
      };

      textarea.addEventListener("input", () => {
        const before = textarea.value.slice(0, textarea.selectionStart);
        const match = before.match(/(^|\s)@([\w.\-]*)$/);
        if (!match) return hide();
        const token = match[2].toLowerCase();
        items = fileNames().filter((n) => n.toLowerCase().includes(token));
        active = 0;
        render();
      });

      textarea.addEventListener(
        "keydown",
        (e) => {
          if (!items.length) return;
          if (e.key === "ArrowDown" || e.key === "ArrowUp") {
            e.preventDefault();
            e.stopImmediatePropagation();
            active =
              (active + (e.key === "ArrowDown" ? 1 : items.length - 1)) %
              items.length;
            render();
          } else if (e.key === "Enter" || e.key === "Tab") {
            e.preventDefault();
            e.stopImmediatePropagation();
            pick(active);
          } else if (e.key === "Escape") {
            hide();
          }
        },
        true
      );

      textarea.addEventListener("blur", () => setTimeout(hide, 150));
    }
  }

  // -- LaTeX in model text ----------------------------------------------------
  // The server replaces $...$ with <span class="npi-tex" data-tex="base64">
  // BEFORE markdown runs, so markdown can never mangle the TeX. The client
  // decodes and typesets each placeholder exactly - no delimiter scanning.
  // KaTeX styles in <head> do not pierce shadow roots, so the stylesheet is
  // fetched once, font URLs made absolute, and adopted where math appears.

  let katexSheet = null;
  let katexSheetLoading = false;

  function loadKatexSheet() {
    if (katexSheet || katexSheetLoading) return;
    katexSheetLoading = true;
    const base = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/";
    fetch(base + "katex.min.css")
      .then((r) => r.text())
      .then((css) => {
        const sheet = new CSSStyleSheet();
        sheet.replaceSync(css.replaceAll("url(fonts/", "url(" + base + "fonts/"));
        katexSheet = sheet;
      })
      .catch(() => (katexSheetLoading = false));
  }

  function renderMath() {
    if (!window.katex) return;
    loadKatexSheet();
    if (!katexSheet) return;
    for (const host of walk(document)) {
      const root = host.shadowRoot;
      if (!root) continue;
      const spans = root.querySelectorAll(".npi-tex:not([data-npi-done])");
      if (!spans.length) continue;
      if (!root.adoptedStyleSheets.includes(katexSheet)) {
        root.adoptedStyleSheets = [...root.adoptedStyleSheets, katexSheet];
      }
      for (const span of spans) {
        span.dataset.npiDone = "1";
        const tex = atob(span.dataset.tex || "");
        try {
          katex.render(tex, span, {
            throwOnError: false,
            displayMode: span.dataset.display === "1",
          });
        } catch (e) {
          span.textContent = "$" + tex + "$";
        }
      }
    }
  }

  setInterval(() => {
    attachSortable();
    attachMention();
    renderMath();
  }, 800);
})();

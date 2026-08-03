/* CGX site behaviour: surface tabs, copy-to-clipboard, scroll reveal.
   No dependencies, no build step -- the page is served as static files. */

(function () {
  "use strict";

  // ------------------------------------------------------------ surface tabs
  var tablist = document.querySelector('[role="tablist"]');
  if (tablist) {
    var tabs = Array.prototype.slice.call(tablist.querySelectorAll('[role="tab"]'));

    function select(tab) {
      tabs.forEach(function (t) {
        var on = t === tab;
        t.setAttribute("aria-selected", on ? "true" : "false");
        var panel = document.getElementById(t.getAttribute("aria-controls"));
        if (panel) panel.hidden = !on;
      });
    }

    tabs.forEach(function (tab, i) {
      tab.addEventListener("click", function () {
        select(tab);
      });
      tab.addEventListener("keydown", function (e) {
        var delta = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
        if (!delta) return;
        e.preventDefault();
        var next = tabs[(i + delta + tabs.length) % tabs.length];
        select(next);
        next.focus();
      });
    });
  }

  // ----------------------------------------------------------------- copy
  document.querySelectorAll("button.copy").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var text = btn.getAttribute("data-copy");
      if (!text || !navigator.clipboard) return;
      navigator.clipboard.writeText(text).then(function () {
        var label = btn.textContent;
        btn.textContent = "copied";
        btn.setAttribute("data-copied", "1");
        setTimeout(function () {
          btn.textContent = label;
          btn.removeAttribute("data-copied");
        }, 1600);
      });
    });
  });

  // ----------------------------------------------------------------- reveal
  var targets = document.querySelectorAll(".reveal");
  var hasIO = "IntersectionObserver" in window;
  if (!hasIO) {
    targets.forEach(function (el) {
      el.classList.add("is-in");
    });
  } else {
    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          entry.target.classList.add("is-in");
          io.unobserve(entry.target);
        });
      },
      { rootMargin: "0px 0px -12% 0px", threshold: 0.08 },
    );
    targets.forEach(function (el) {
      io.observe(el);
    });
  }

  // ------------------------------------------------ scroll progress + to-top
  var progress = document.getElementById("progress");
  var toTop = document.querySelector(".to-top");

  function onScroll() {
    var doc = document.documentElement;
    var top = doc.scrollTop || document.body.scrollTop;
    var height = doc.scrollHeight - doc.clientHeight;
    if (progress) progress.style.width = (height > 0 ? (top / height) * 100 : 0) + "%";
    if (toTop) toTop.classList.toggle("is-visible", top > 640);
  }

  var ticking = false;
  window.addEventListener(
    "scroll",
    function () {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(function () {
        onScroll();
        ticking = false;
      });
    },
    { passive: true },
  );
  onScroll();

  // ------------------------------------------------------------- scroll spy
  var spySections = Array.prototype.slice.call(document.querySelectorAll("main > section[id]"));
  var navLinks = Array.prototype.slice.call(
    document.querySelectorAll('.topnav a[href^="#"], .railnav a[href^="#"]'),
  );

  if (hasIO && spySections.length && navLinks.length) {
    var spy = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          var id = entry.target.id;
          navLinks.forEach(function (link) {
            link.classList.toggle("is-active", link.getAttribute("href") === "#" + id);
          });
        });
      },
      { rootMargin: "-45% 0px -50% 0px", threshold: 0 },
    );
    spySections.forEach(function (s) {
      spy.observe(s);
    });
  }
})();

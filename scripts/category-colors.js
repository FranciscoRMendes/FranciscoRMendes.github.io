'use strict';

// Inject a small script that reads each post card's category link and sets
// a data-category attribute, which CSS uses for the left-border tint.
hexo.extend.filter.register('after_render:html', function(html) {
  const script = `<script>
(function () {
  function applyCategories() {
    document.querySelectorAll('.card').forEach(function (card) {
      var link = card.querySelector('a[href*="/categories/"]');
      if (!link) return;
      var parts = link.getAttribute('href').split('/').filter(Boolean);
      var cat = parts[parts.length - 1];
      if (cat) card.dataset.category = cat;
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', applyCategories);
  } else {
    applyCategories();
  }
})();
</script>`;
  return html.replace('</body>', script + '</body>');
}, 20);

'use strict';

// Note: this must run on 'after_render:html', not 'after_post_render'.
// after_post_render fires while posts are still being loaded one at a time,
// so hexo.locals.get('posts') is an incomplete snapshot at that point and
// the "2+ posts in this series" check silently never passes. By the time
// after_render:html fires for a post's route, all posts are loaded.
hexo.extend.filter.register('after_render:html', function(html, data) {
  if (!data || !data.path) return html;

  const allPosts = hexo.locals.get('posts');
  if (!allPosts) return html;

  const posts = allPosts.toArray ? allPosts.toArray() : allPosts;
  const post = posts.find(p => p.path === data.path || data.path.indexOf(p.path) === 0);
  if (!post || !post.series) return html;

  const seriesName = post.series;

  const seriesPosts = posts
    .filter(p => p.series === seriesName)
    .sort((a, b) => {
      const ai = a.series_index != null ? a.series_index : 9999;
      const bi = b.series_index != null ? b.series_index : 9999;
      if (ai !== bi) return ai - bi;
      return a.date - b.date;
    });

  if (seriesPosts.length < 2) return html;

  const items = seriesPosts.map(p => {
    const isCurrent = p.path === post.path;
    if (isCurrent) {
      return `<li class="series-item series-current"><span>${p.title}</span></li>`;
    }
    return `<li class="series-item"><a href="${hexo.config.root}${p.path}">${p.title}</a></li>`;
  }).join('\n');

  const box = `<div class="series-box">
  <div class="series-label">Series</div>
  <div class="series-name">${seriesName}</div>
  <ol class="series-list">${items}</ol>
</div>`;

  return html.replace('<div class="content">', '<div class="content">' + box);
});

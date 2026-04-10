'use strict';

hexo.extend.filter.register('after_post_render', function(data) {
  if (!data.series) return data;

  let allPosts;
  try {
    allPosts = hexo.locals.get('posts');
  } catch (e) {
    return data;
  }
  if (!allPosts) return data;

  const seriesName = data.series;

  const posts = allPosts
    .filter(p => p.series === seriesName)
    .sort((a, b) => {
      const ai = a.series_index != null ? a.series_index : 9999;
      const bi = b.series_index != null ? b.series_index : 9999;
      if (ai !== bi) return ai - bi;
      return a.date - b.date;
    });

  if (posts.length < 2) return data;

  const items = posts.map(p => {
    const isCurrent = p.path === data.path;
    if (isCurrent) {
      return `<li class="series-item series-current"><span>${p.title}</span></li>`;
    }
    return `<li class="series-item"><a href="${hexo.config.root}${p.path}">${p.title}</a></li>`;
  }).join('\n');

  const box = `<div class="series-box">
  <div class="series-label">Series</div>
  <div class="series-name">${seriesName}</div>
  <ol class="series-list">${items}</ol>
</div>
`;

  data.content = box + data.content;
  return data;
});

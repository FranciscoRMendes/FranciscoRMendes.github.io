'use strict';

hexo.extend.generator.register('series-index', function(locals) {
  const posts = locals.posts.toArray();

  const seriesMap = {};
  posts.forEach(post => {
    if (!post.series) return;
    if (!seriesMap[post.series]) seriesMap[post.series] = [];
    seriesMap[post.series].push(post);
  });

  Object.keys(seriesMap).forEach(name => {
    seriesMap[name].sort((a, b) => {
      const ai = a.series_index != null ? a.series_index : 9999;
      const bi = b.series_index != null ? b.series_index : 9999;
      return ai - bi;
    });
  });

  const boxes = Object.keys(seriesMap).sort().map(name => {
    const items = seriesMap[name].map(post => {
      return `<li class="series-item"><a href="${hexo.config.root}${post.path}">${post.title}</a></li>`;
    }).join('\n');

    return `<div class="series-box">
  <div class="series-label">Series</div>
  <div class="series-name">${name}</div>
  <ol class="series-list">${items}</ol>
</div>`;
  }).join('\n');

  const content = `<div class="series-index">${boxes}</div>`;

  return {
    path: 'series/index.html',
    layout: ['page'],
    data: {
      title: 'Series',
      content
    }
  };
});

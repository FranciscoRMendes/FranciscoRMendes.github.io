'use strict';

// Inject custom CSS link into every generated HTML page's <head>
hexo.extend.filter.register('after_render:html', function(html) {
    return html.replace(
        '</head>',
        '<link rel="stylesheet" href="/css/custom.css"></head>'
    );
});

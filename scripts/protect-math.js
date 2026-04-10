'use strict';

// Protect math blocks from hexo-renderer-marked before markdown processing.
//
// Problem: hexo-renderer-marked processes underscores inside $$...$$ as
// markdown emphasis, breaking LaTeX subscripts like \hat{W}_{16}.
//
// Historical convention: source files write \\\\ (4 backslashes) so that
// after markdown's \\ → \ conversion, MathJax receives \\ (LaTeX line break).
// Since we bypass markdown for math blocks, we normalise \\\\ → \\ ourselves.
//
// Pipeline:
// 1. before_post_render (priority 1): replace math blocks with HTML comment
//    placeholders so marked never sees the LaTeX. Also normalise \\\\ → \\.
// 2. after_post_render (priority 4): restore math blocks into the HTML output,
//    then clear `data.mathjax` so hexo-filter-mathjax (priority 5) skips this
//    post. Icarus's global MathJax CDN injection handles client-side rendering.

const mathCache = new Map();

hexo.extend.filter.register('before_post_render', function(data) {
  const key = data.full_source || data.path;
  if (!key) return data;

  const blocks = [];

  function capture(match) {
    // Normalise \\\\ → \\ because we bypass markdown's own \\ → \ conversion.
    const normalised = match.replace(/\\\\\\\\/g, '\\\\');
    blocks.push(normalised);
    return `<!--HEXOMATH${blocks.length - 1}-->`;
  }

  // Protect display math $$...$$ first (must precede inline to avoid matching
  // the $$ delimiters as two separate $…$ inline blocks).
  data.content = data.content.replace(/\$\$([\s\S]*?)\$\$/g, capture);

  // Protect inline math $...$ (single-line, not $$).
  data.content = data.content.replace(/(?<!\$)\$([^\n$]+?)\$(?!\$)/g, capture);

  if (blocks.length > 0) {
    mathCache.set(key, blocks);
  }

  return data;
}, 1);

hexo.extend.filter.register('after_post_render', function(data) {
  const key = data.full_source || data.path;
  if (!key) return data;

  const blocks = mathCache.get(key);
  if (blocks) {
    data.content = data.content.replace(/<!--HEXOMATH(\d+)-->/g, (_, idx) => {
      return blocks[parseInt(idx, 10)] || '';
    });
    mathCache.delete(key);
  }

  // Prevent hexo-filter-mathjax (after_post_render priority 5) from running.
  // It has a "Can't find handler for document" crash on some posts and would
  // conflict with Icarus's global client-side MathJax injection.
  data.mathjax = false;

  return data;
}, 4);

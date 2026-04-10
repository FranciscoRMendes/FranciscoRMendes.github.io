'use strict';

hexo.extend.filter.register('after_render:html', function(html) {
    // 1. Anti-FOUC — set 'dark' class on <html> before any CSS renders
    const antiFlash = '<script>' +
        '(function(){' +
        'try{' +
        'var s=localStorage.getItem(\'dark-mode\');' +
        'if(s===\'true\'||(s===null&&window.matchMedia(\'(prefers-color-scheme:dark)\').matches)){' +
        'document.documentElement.classList.add(\'dark\');' +
        '}' +
        '}catch(e){}' +
        '})();' +
        '</script>';

    // 2. Toggle button — inserted into the navbar after the GitHub icon link
    const toggleBtn =
        '<a id="dark-mode-toggle" class="navbar-item" role="button" ' +
        'aria-label="Toggle dark mode" title="Toggle dark mode">' +
        '<span class="icon is-small"><i class="fas fa-moon"></i></span>' +
        '</a>';

    // 3. Toggle logic — syncs icon and persists preference
    const toggleScript = '<script>' +
        '(function(){' +
        'var btn=document.getElementById(\'dark-mode-toggle\');' +
        'if(!btn)return;' +
        'var icon=btn.querySelector(\'i\');' +
        'function sync(){' +
        'var dark=document.documentElement.classList.contains(\'dark\');' +
        'icon.className=dark?\'fas fa-sun\':\'fas fa-moon\';' +
        'btn.title=dark?\'Switch to light mode\':\'Switch to dark mode\';' +
        '}' +
        'sync();' +
        'btn.addEventListener(\'click\',function(){' +
        'var isDark=document.documentElement.classList.toggle(\'dark\');' +
        'localStorage.setItem(\'dark-mode\',String(isDark));' +
        'sync();' +
        '});' +
        '})();' +
        '</script>';

    // Inject anti-FOUC immediately after <meta charset> so it runs
    // before any stylesheet is parsed — prevents the light-flash on load.
    html = html.replace(/(<meta\s+charset[^>]*>)/i, '$1' + antiFlash);

    // Inject toggle button right after the GitHub navbar link
    html = html.replace(
        /(href="https:\/\/github\.com\/FranciscoRMendes"[^>]*>[\s\S]*?<\/a>)/,
        '$1' + toggleBtn
    );

    // Inject toggle logic just before </body>
    html = html.replace('</body>', toggleScript + '</body>');

    return html;
});

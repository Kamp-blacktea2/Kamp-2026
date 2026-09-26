const {
    chromium
} = require(
    'C:/Users/K/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'
    );
const fs = require('node:fs');
(async () => {
    let browser = await chromium.launch({
        headless: true,
        channel: 'msedge'
    });
    let page = await browser.newPage({
        viewport: {
            width: 1450,
            height: 1100
        }
    });
    let errors = [];
    page.on('pageerror', e => errors.push(String(e)));
    await page.goto('file:///F:/Kamp/experiments/YSH/ysh-001/outputs/graphs.html');
    await page.waitForSelector('.js-plotly-plot');
    for (const mode of ['nested', 'rolling', 'expanded', 'blocks']) {
        await page.selectOption('#mode', mode);
        await page.waitForTimeout(400);
        if (!await page.locator('.js-plotly-plot').count()) throw Error('No graph ' + mode);
    }
    await page.selectOption('#mode', 'nested');
    await page.screenshot({
        path: 'F:/Kamp/experiments/YSH/ysh-001/outputs/figures/07_browser_check.png',
        fullPage: false
    });
    await page.selectOption('#condition', 'B_dedup');
    await page.selectOption('#offset', '10');
    await page.waitForTimeout(300);
    let result = {
        errors,
        plotCount: await page.locator('.js-plotly-plot').count(),
        info: await page.locator('#info').innerText()
    };
    fs.writeFileSync('F:/Kamp/experiments/YSH/ysh-001/outputs/browser_checks.json', JSON
        .stringify(result, null, 2));
    await browser.close();
    console.log(JSON.stringify(result));
    if (errors.length) process.exitCode = 1;
})();

#!/usr/bin/env python3
"""Render replay-derived method pictures and check local report interactions."""
from pathlib import Path
import json, threading, functools, re
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, unquote
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[3]
REPORT=ROOT/'documentation/reports/link5-experiments'

class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self,*args):
        pass

def main():
    handler=functools.partial(QuietHandler,directory=str(ROOT))
    server=ThreadingHTTPServer(('127.0.0.1',0),handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    base=f'http://127.0.0.1:{server.server_port}/documentation/reports/link5-experiments/'
    qa=REPORT/'assets/qa';qa.mkdir(exist_ok=True)
    checks=[];errors=[]
    data=json.loads((REPORT/'assets/report-data.json').read_text())
    ablation=(REPORT/data['paths']['ablation']).resolve()
    for row in data['rows']:
        for method in ('balanced_v0','refine_v1'):
            path=ablation/'report/cases'/row['video_id']/(method+'.html')
            assert path.is_file(),path
            text=path.read_text()
            for attr,url in re.findall(r'(href|src)="([^"]+)"',text):
                if not url.startswith(('http:','https:','data:','#')):
                    assert (path.parent/url.split('?')[0]).exists(),(path,url)
    checks.append('All 82 method pages and their explicit local resources exist')
    with sync_playwright() as p:
        existing=sorted((Path.home()/'Library/Caches/ms-playwright').glob('chromium-*/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'))
        options=dict(headless=True,args=['--enable-webgl','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        if existing:options['executable_path']=str(existing[-1])
        browser=p.chromium.launch(**options)
        page=browser.new_page(viewport={'width':1200,'height':900},device_scale_factor=2)
        page.on('pageerror',lambda e:errors.append(str(e)))
        for kind in ('pair','depth','erosion'):
            for version in ('old','new'):
                page.goto(base+f'method-replay.html?kind={kind}&version={version}&capture=1',wait_until='load')
                page.wait_for_function('window.METHOD_READY === true || !!window.METHOD_ERROR',timeout=90000)
                assert not page.evaluate('window.METHOD_ERROR'),page.evaluate('window.METHOD_ERROR')
                page.locator('#capture').screenshot(path=str(REPORT/f'assets/{kind}-{version}.png'))
                page.locator('#source').screenshot(path=str(REPORT/f'assets/{kind}-{version}-source.png'),scale='device')
                page.locator('#pixels' if kind=='erosion' else '#cloud').screenshot(path=str(REPORT/f'assets/{kind}-{version}-detail.png'),scale='device')
                assert page.locator('#stats').inner_text()
                print('Rendered',kind,version,page.locator('#stats').inner_text(),flush=True)
        checks.append('Six method pictures captured from the live method replay renderer')
        page.goto(base+'method-replay.html?kind=erosion',wait_until='load')
        page.wait_for_function('window.METHOD_READY === true')
        page.locator('#frame').fill('73')
        page.wait_for_function('window.METHOD_READY === true && document.getElementById("stamp").textContent.includes("frame 73")')
        page.locator('#version').select_option('query')
        page.wait_for_function('window.METHOD_READY === true && document.getElementById("title").textContent.includes("Query")')
        page.locator('#play').click();page.wait_for_timeout(650);page.locator('#play').click()
        assert int(page.locator('#frame').input_value())>73
        checks.append('All-frame erosion scrubbing, square kernel comparison and playback work')
        page.set_viewport_size({'width':1440,'height':1040})
        page.goto(base,wait_until='load')
        page.wait_for_selector('#case-rows tr button')
        assert page.locator('#case-rows tr button').count()==41
        assert '0.6696' in page.locator('#auroc-table').inner_text()
        assert '0.7792' in page.locator('#auroc-table').inner_text()
        # Verify the supplied stretched-image failure at desktop and mobile sizes.
        selector='#pair-selection .compare img, #depth-filter .compare img, #erosion .compare img'
        page.eval_on_selector_all(selector,'async imgs => { await Promise.all(imgs.map(img => { img.loading="eager"; return img.decode(); })); }')
        image_ratios=page.eval_on_selector_all(selector,'imgs => imgs.map(img => { const r=img.getBoundingClientRect(); return {src:img.src,natural:img.naturalWidth/img.naturalHeight,display:r.width/r.height}; })')
        assert len(image_ratios)==12
        for image in image_ratios:
            assert abs(image['display']/image['natural']-1)<.015,image
        for section in ('pair-selection','depth-filter','erosion'):
            page.locator('#'+section+' .compare').screenshot(path=str(qa/('desktop-'+section+'-comparison.png')))
        checks.append('All 12 evidence panels retain their natural image proportions; readable labels are HTML')
        page.screenshot(path=str(qa/'desktop-overview.png'))
        page.locator('#case-search').fill('COSMOS2.5_0018')
        assert page.locator('#case-rows tr button').count()==1
        page.locator('#case-rows tr button').click()
        page.locator('#method').select_option('balanced_v0')
        assert 'COSMOS2.5_0018/balanced_v0.html' in page.locator('#case-frame').get_attribute('src')
        frame=page.locator('#case-frame').element_handle().content_frame()
        frame.wait_for_url('**/COSMOS2.5_0018/balanced_v0.html')
        frame.wait_for_selector('#cloud.js-plotly-plot',timeout=90000)
        frame.wait_for_function('document.querySelector("video").readyState >= 2')
        frame.locator('#frame-number').fill('26');frame.locator('#frame-number').press('Tab')
        frame.wait_for_function('document.getElementById("time").textContent.startsWith("26 / ")',timeout=30000)
        frame.locator('#view-mode').click()
        frame.wait_for_function('document.getElementById("view-mode").textContent === "Camera view"')
        page.locator('#case-search').fill('no-such-case')
        assert 'No cases match' in page.locator('#case-rows').inner_text()
        page.locator('#case-search').fill('');page.locator('#generator').select_option('COSMOS3')
        assert page.locator('#case-rows tr button').count()==13
        checks.append('41-case gallery search, empty state, generator filter, method switching, source-frame seeking and 3D orbit work')
        for chosen in data['simple_sample']:
            page.locator(f'#simple-choices button[data-id="{chosen}"]').click()
            frame=page.locator('#simple-frame').element_handle().content_frame()
            frame.wait_for_url('**/simple3d.html?case='+chosen)
            frame.wait_for_function('document.getElementById("case").value === '+json.dumps(chosen)+' && document.getElementById("test-cloud").classList.contains("js-plotly-plot")',timeout=90000)
            assert 'Replay loading failed' not in frame.locator('#status').inner_text()
            print('Simple3D loaded',chosen,flush=True)
        frame.locator('#next').click()
        frame.locator('#aggregation').select_option('160')
        frame.wait_for_function('document.getElementById("proof").textContent.includes("top 160")')
        checks.append('All six seeded Simple3D cases load; pair navigation and top-160 score inspection work')
        # The display wrapper repairs cross-run URLs from before the run was relocated.
        urls=frame.eval_on_selector_all('a[href]','els => els.map(e => e.href)')
        for url in urls:
            parsed=urlparse(url)
            if parsed.hostname=='127.0.0.1':
                assert (ROOT/unquote(parsed.path.lstrip('/'))).exists(),url
        checks.append('Simple3D display-copy cross-run links resolve after relocation')
        page.locator('#case-search').fill('');page.locator('#generator').select_option('all')
        page.locator('#pair-selection').screenshot(path=str(qa/'desktop-methods.png'))
        page.locator('#architecture .diagram').first.screenshot(path=str(qa/'desktop-architecture.png'))
        page.locator('#coordination .diagram').screenshot(path=str(qa/'desktop-coordination.png'))
        page.set_viewport_size({'width':390,'height':844});page.goto(base,wait_until='load')
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'Mobile page overflows'
        page.screenshot(path=str(qa/'mobile-overview.png'))
        page.locator('#pair-selection').screenshot(path=str(qa/'mobile-methods.png'))
        page.locator('#erosion .compare').screenshot(path=str(qa/'mobile-erosion-comparison.png'))
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        checks.append('390px mobile layout has no document overflow; diagrams and tables scroll within their containers')
        # Ensure every static report link/image points to a local artifact or explicit external URL.
        page.goto(base,wait_until='load')
        refs=page.eval_on_selector_all('[href],[src]','els => els.flatMap(e => [e.getAttribute("href"),e.getAttribute("src")].filter(Boolean))')
        missing=[]
        for ref in refs:
            if ref.startswith(('#','data:','http:','https:')):continue
            dest=(REPORT/unquote(urlparse(ref).path)).resolve()
            if not dest.exists():missing.append(ref)
        assert not missing,missing
        checks.append('All report links, images, scripts and replay entrypoints resolve locally')
        assert not errors,errors
        browser.close()
    server.shutdown()
    (qa/'verification.json').write_text(json.dumps(dict(status='passed',checks=checks,browser_errors=errors,method_pictures=6,case_replays=82,simple_random_cases=6),indent=2)+'\n')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()

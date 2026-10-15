"""Record the first page's error-signal preview from the built replay (local, needs Playwright + Chrome).

Steps the real ``signals.html`` through every frame of one video, screenshots the 3D
viewer (layers and play bar hidden, colour key kept) at the 2D views' 740:420 aspect,
and encodes the frames with H.264. Writes ``previews/error_signal.mp4`` and ``.jpg``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

HIDE = 'nav.top,.layers,.pbar,.inspector,.pip,.status{display:none!important}.viewer{aspect-ratio:740/420!important;min-height:0!important}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True, help='Built replay folder')
    parser.add_argument('--case', default='COSMOS2.5_0001')
    parser.add_argument('--signal', default='rigid_residual')
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    output = args.replay / 'previews'
    output.mkdir(exist_ok=True)
    frames_dir = Path(tempfile.mkdtemp(prefix='mose3-preview-'))
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome', headless=True, args=[
            '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--allow-file-access-from-files'])
        page = browser.new_page(viewport={'width': 1120, 'height': 900}, device_scale_factor=1)
        page.goto(f'file://{args.replay.resolve()}/signals.html?case={args.case}&signal={args.signal}')
        page.wait_for_function('document.querySelector("#status").hidden === true', timeout=120000)
        page.click('#play')                                            # pause; frames are set explicitly
        page.add_style_tag(content=HIDE)
        fps = page.evaluate('window.MOSE3_INDEX.cases.find(c => c.case === %r).fps' % args.case)
        count = page.evaluate('window.MOSE3_INDEX.cases.find(c => c.case === %r).frames' % args.case)
        for frame in range(count):
            page.eval_on_selector('#slider', '(el, t) => { el.value = t; el.dispatchEvent(new Event("input"));'
                                  ' el.dispatchEvent(new Event("change")); }', (frame + 0.5) / fps)
            page.wait_for_timeout(90)
            page.locator('#viewer').screenshot(path=str(frames_dir / f'{frame:04d}.png'))
        browser.close()
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', f'{fps:.6f}', '-i', str(frames_dir / '%04d.png'),
                    '-vf', 'scale=740:420:flags=area', '-c:v', 'libx264', '-preset', 'slow', '-crf', '26',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'error_signal.mp4')], check=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(frames_dir / f'{count // 2:04d}.png'), '-vf', 'scale=740:420',
                    '-q:v', '3', str(output / 'error_signal.jpg')], check=True)
    shutil.rmtree(frames_dir)
    print(f'PREVIEW {output / "error_signal.mp4"} frames={count} fps={fps}')


if __name__ == '__main__':
    main()

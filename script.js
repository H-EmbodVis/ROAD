const showcaseItems = [
  { id: '001', title: 'ROAD027 · Bread man', condition: './assets/showcase/001/condition.webp', preview: './assets/showcase/001/preview.webp', model: './assets/showcase/001/model.glb' },
  { id: '002', title: 'Example 002', condition: './assets/showcase/002/condition.webp', preview: './assets/showcase/002/preview.webp', model: './assets/showcase/002/model.glb' },
  { id: '003', title: 'Example 003', condition: './assets/showcase/003/condition.webp', preview: './assets/showcase/003/preview.webp', model: './assets/showcase/003/model.glb' },
  { id: '004', title: 'Example 004', condition: './assets/showcase/004/condition.webp', preview: './assets/showcase/004/preview.webp', model: './assets/showcase/004/model.glb' },
  { id: '005', title: 'Example 005', condition: './assets/showcase/005/condition.webp', preview: './assets/showcase/005/preview.webp', model: './assets/showcase/005/model.glb' },
  { id: '006', title: 'Example 006', condition: './assets/showcase/006/condition.webp', preview: './assets/showcase/006/preview.webp', model: './assets/showcase/006/model.glb' }
];

const viewer = document.getElementById('modelViewer');
const conditionImage = document.getElementById('conditionImage');
const selectedTitle = document.getElementById('selectedTitle');
const selectedIndex = document.getElementById('selectedIndex');
const downloadModel = document.getElementById('downloadModel');
const thumbs = document.getElementById('galleryThumbs');

function selectItem(index) {
  const item = showcaseItems[index];
  if (!item || !viewer) return;
  viewer.setAttribute('poster', item.preview);
  viewer.setAttribute('src', item.model);
  conditionImage.src = item.condition;
  conditionImage.alt = `Conditioning image for ${item.title}`;
  selectedTitle.textContent = item.title;
  selectedIndex.textContent = `${String(index + 1).padStart(2, '0')} / ${String(showcaseItems.length).padStart(2, '0')}`;
  downloadModel.href = item.model;
  downloadModel.setAttribute('download', `ROAD-example-${item.id}.glb`);
  [...thumbs.children].forEach((button, i) => button.setAttribute('aria-current', i === index ? 'true' : 'false'));
}

showcaseItems.forEach((item, index) => {
  const button = document.createElement('button');
  button.className = 'gallery-thumb';
  button.type = 'button';
  button.setAttribute('aria-label', `Show ${item.title}`);
  button.innerHTML = `<img src="${item.preview}" alt="Generated 3D preview ${item.id}" loading="lazy"><span>${item.id}</span>`;
  button.addEventListener('click', () => selectItem(index));
  thumbs?.appendChild(button);
});
selectItem(0);

viewer?.addEventListener('progress', event => {
  const fill = viewer.querySelector('.viewer-progress-fill');
  if (fill) fill.style.width = `${event.detail.totalProgress * 100}%`;
});

async function copyText(targetId, button) {
  const target = document.getElementById(targetId);
  if (!target) return;
  try {
    await navigator.clipboard.writeText(target.textContent);
    const oldText = button.textContent;
    button.textContent = 'Copied';
    setTimeout(() => { button.textContent = oldText; }, 1500);
  } catch (_) {
    button.textContent = 'Select text';
  }
}

document.querySelectorAll('.copy-code').forEach(button => {
  button.addEventListener('click', () => copyText(button.dataset.copy, button));
});

document.getElementById('copyCitation')?.addEventListener('click', event => {
  copyText('citationText', event.currentTarget);
});

const dialog = document.getElementById('imageDialog');
const dialogImage = dialog?.querySelector('img');
document.querySelectorAll('.zoomable img').forEach(image => {
  image.addEventListener('click', () => {
    if (!dialog || !dialogImage) return;
    dialogImage.src = image.src;
    dialogImage.alt = image.alt;
    dialog.showModal();
  });
});
dialog?.querySelector('button')?.addEventListener('click', () => dialog.close());
dialog?.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });

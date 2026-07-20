const showcaseItems = [
  {
    "id": "001",
    "condition": "./assets/showcase/001/condition.webp",
    "preview": "./assets/showcase/001/preview.webp",
    "model": "./assets/showcase/001/model.glb"
  },
  {
    "id": "002",
    "condition": "./assets/showcase/002/condition.webp",
    "preview": "./assets/showcase/002/preview.webp",
    "model": "./assets/showcase/002/model.glb"
  },
  {
    "id": "003",
    "condition": "./assets/showcase/003/condition.webp",
    "preview": "./assets/showcase/003/preview.webp",
    "model": "./assets/showcase/003/model.glb"
  },
  {
    "id": "004",
    "condition": "./assets/showcase/004/condition.webp",
    "preview": "./assets/showcase/004/preview.webp",
    "model": "./assets/showcase/004/model.glb"
  },
  {
    "id": "005",
    "condition": "./assets/showcase/005/condition.webp",
    "preview": "./assets/showcase/005/preview.webp",
    "model": "./assets/showcase/005/model.glb"
  },
  {
    "id": "006",
    "condition": "./assets/showcase/006/condition.webp",
    "preview": "./assets/showcase/006/preview.webp",
    "model": "./assets/showcase/006/model.glb"
  }
];

const viewer = document.getElementById('modelViewer');
const conditionImage = document.getElementById('conditionImage');
const selectedTitle = document.getElementById('selectedTitle');
const selectedIndex = document.getElementById('selectedIndex');
const downloadModel = document.getElementById('downloadModel');
const thumbs = document.getElementById('galleryThumbs');

function selectItem(index) {
  const item = showcaseItems[index];
  viewer.setAttribute('poster', item.preview);
  viewer.setAttribute('src', item.model);
  conditionImage.src = item.condition;
  conditionImage.alt = `Conditioning image for example ${item.id}`;
  selectedTitle.textContent = `Example ${item.id}`;
  selectedIndex.textContent = `${String(index + 1).padStart(2, '0')} / ${String(showcaseItems.length).padStart(2, '0')}`;
  downloadModel.href = item.model;
  downloadModel.setAttribute('download', `ROAD-example-${item.id}.glb`);
  [...thumbs.children].forEach((button, i) => button.setAttribute('aria-current', i === index ? 'true' : 'false'));
}

showcaseItems.forEach((item, index) => {
  const button = document.createElement('button');
  button.className = 'gallery-thumb';
  button.type = 'button';
  button.setAttribute('aria-label', `Show example ${item.id}`);
  button.innerHTML = `<img src="${item.preview}" alt="Generated 3D preview ${item.id}" loading="lazy"><span>${item.id}</span>`;
  button.addEventListener('click', () => selectItem(index));
  thumbs.appendChild(button);
});
selectItem(0);

viewer?.addEventListener('progress', event => {
  const fill = viewer.querySelector('.viewer-progress-fill');
  if (fill) fill.style.width = `${event.detail.totalProgress * 100}%`;
});

const copyButton = document.getElementById('copyCitation');
copyButton?.addEventListener('click', async () => {
  const text = document.getElementById('citationText').textContent;
  try {
    await navigator.clipboard.writeText(text);
    copyButton.textContent = 'Copied';
    setTimeout(() => copyButton.textContent = 'Copy', 1500);
  } catch (_) {
    copyButton.textContent = 'Select text';
  }
});

const dialog = document.getElementById('imageDialog');
const dialogImage = dialog?.querySelector('img');
document.querySelectorAll('.zoomable img').forEach(image => {
  image.addEventListener('click', () => {
    dialogImage.src = image.src;
    dialogImage.alt = image.alt;
    dialog.showModal();
  });
});
dialog?.querySelector('button').addEventListener('click', () => dialog.close());
dialog?.addEventListener('click', event => {
  if (event.target === dialog) dialog.close();
});

async function copyText(targetId, button) {
  const target = document.getElementById(targetId);
  if (!target) return;
  try {
    await navigator.clipboard.writeText(target.textContent);
    const original = button.textContent;
    button.textContent = 'Copied';
    window.setTimeout(() => { button.textContent = original; }, 1400);
  } catch (_) {
    button.textContent = 'Select text';
  }
}

document.querySelectorAll('.copy-code').forEach(button => {
  button.addEventListener('click', () => copyText(button.dataset.copy, button));
});

const dialog = document.getElementById('imageDialog');
const dialogImage = dialog?.querySelector('img');
document.querySelectorAll('.zoomable img').forEach(image => {
  image.addEventListener('click', () => {
    if (!dialog || !dialogImage) return;
    dialogImage.src = image.currentSrc || image.src;
    dialogImage.alt = image.alt;
    dialog.showModal();
  });
});
dialog?.querySelector('button')?.addEventListener('click', () => dialog.close());
dialog?.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });

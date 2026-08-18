document.addEventListener('DOMContentLoaded', () => {
  const progress = document.createElement('div');
  progress.id = 'scroll-progress';
  document.body.appendChild(progress);

  const updateProgress = () => {
    const root = document.documentElement;
    const max = root.scrollHeight - root.clientHeight;
    progress.style.width = `${max > 0 ? (root.scrollTop / max) * 100 : 0}%`;
  };
  document.addEventListener('scroll', updateProgress, {passive: true});
  updateProgress();

  document.querySelectorAll('table').forEach(table => {
    const wrap = document.createElement('div');
    wrap.className = 'table-wrap';
    table.parentNode.insertBefore(wrap, table);
    wrap.appendChild(table);
  });

  document.querySelectorAll('pre').forEach(pre => {
    const button = document.createElement('button');
    button.className = 'copy-code';
    button.type = 'button';
    button.textContent = 'copy';
    button.addEventListener('click', async () => {
      await navigator.clipboard.writeText(pre.innerText.replace(/^copy\n/, ''));
      button.textContent = 'copied';
      setTimeout(() => button.textContent = 'copy', 900);
    });
    pre.appendChild(button);
  });
});

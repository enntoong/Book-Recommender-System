(() => {
  document.querySelectorAll('[data-toast]').forEach((toast) => {
    window.setTimeout(() => {
      toast.classList.add('hiding');
      window.setTimeout(() => toast.remove(), 300);
    }, 3200);
  });

  let controller = null;
  document.querySelectorAll('[data-autocomplete-form]').forEach((form) => {
    const input = form.querySelector('[data-autocomplete-input]');
    const results = form.querySelector('[data-autocomplete-results]');
    if (!input || !results) return;

    let timer;
    input.addEventListener('input', () => {
      clearTimeout(timer);
      const q = input.value.trim();
      if (q.length < 2) {
        results.hidden = true;
        results.innerHTML = '';
        return;
      }
      timer = setTimeout(async () => {
        if (controller) controller.abort();
        controller = new AbortController();
        try {
          const response = await fetch(`/api/search?q=${encodeURIComponent(q)}`, { signal: controller.signal });
          if (!response.ok) return;
          const books = await response.json();
          const back = `${window.location.pathname}${window.location.search}`;
          results.innerHTML = books.map((book) => {
            const url = new URL(book.url, window.location.origin);
            url.searchParams.set('back', back);
            return `
            <a class="autocomplete-item" href="${url.pathname}${url.search}">
              <strong>${escapeHtml(book.title)}</strong>
              <span>${escapeHtml(book.author)}</span>
            </a>`;
          }).join('');
          results.hidden = books.length === 0;
        } catch (error) {
          if (error.name !== 'AbortError') results.hidden = true;
        }
      }, 180);
    });

    document.addEventListener('click', (event) => {
      if (!form.contains(event.target)) results.hidden = true;
    });
  });

  function escapeHtml(value) {
    return String(value)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#039;');
  }
})();
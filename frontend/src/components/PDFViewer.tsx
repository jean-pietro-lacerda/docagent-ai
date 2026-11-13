import { useEffect, useRef } from 'react';
import { Icon } from './Icon';
export function PDFViewer({ url, filename, page, onClose }: { url: string; filename: string; page: number; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => { dialog.current?.showModal(); return () => { URL.revokeObjectURL(url); }; }, [url]);
  return <dialog ref={dialog} className="pdf-dialog" onCancel={onClose} onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
    <header><div><strong>{filename}</strong><span className="muted">Página {page}</span></div><button className="icon-button" aria-label="Fechar visualização" onClick={onClose}><Icon name="close" /></button></header>
    <iframe title={`PDF: ${filename}, página ${page}`} src={`${url}#page=${page}`} />
    <footer>Se o navegador não exibir o PDF, <a href={url} download={filename}>baixe o arquivo</a> e abra na página {page}.</footer>
  </dialog>;
}

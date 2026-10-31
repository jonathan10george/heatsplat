import { useMemo } from "react";
import { marked } from "marked";
import type { Report } from "../types";

export default function ReportModal({ report, onClose }: { report: Report; onClose: () => void }) {
  const html = useMemo(() => marked.parse(report.markdown, { async: false }) as string, [report.markdown]);
  const download = () => {
    const blob = new Blob([report.markdown], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${report.scene_id}-audit.md`;
    a.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000); // Safari can drop the download if revoked at once
  };
  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <div className="modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true" aria-label="Audit report">
        <div className="modal-head">
          <span className="pill">{report.generator}</span>
          <div className="modal-actions">
            <button className="btn ghost" onClick={download}>
              Download .md
            </button>
            <button className="btn ghost" onClick={onClose} aria-label="Close">
              ✕
            </button>
          </div>
        </div>
        <div className="report" dangerouslySetInnerHTML={{ __html: html }} />
      </div>
    </div>
  );
}

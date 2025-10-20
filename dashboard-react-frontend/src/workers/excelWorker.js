// Web Worker para ler e converter a planilha em JSON sem bloquear a UI
import { read, utils } from "xlsx";

self.onmessage = async (e) => {
  try {
    const buf = e.data; // ArrayBuffer
    const wb = read(new Uint8Array(buf), { type: "array" });
    const sheet = wb.Sheets[wb.SheetNames[0]];
    const json = utils.sheet_to_json(sheet, { raw: true });
    self.postMessage({ success: true, data: json });
  } catch (err) {
    self.postMessage({ success: false, error: err?.message || "Erro ao ler planilha" });
  }
};

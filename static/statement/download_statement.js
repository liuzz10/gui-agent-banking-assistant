function downloadStatement(tableId, filename) {
  const table = document.getElementById(tableId);
  const rows = Array.from(table.querySelectorAll("tr"));

  const csv = rows.map(row => {
    const cells = Array.from(row.querySelectorAll("th, td"));
    return cells.map(cell => '"' + cell.innerText.replace(/"/g, '""') + '"').join(",");
  }).join("\n");

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.style.display = "none";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}



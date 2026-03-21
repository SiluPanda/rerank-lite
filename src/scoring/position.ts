export function positionScore(originalRank: number, totalDocs: number): number {
  if (totalDocs <= 0) return 0
  return 1 - originalRank / totalDocs
}

// we want to ignore basically all the punctuation

export function wordCount(text: string): number {
  let n = 0;
  const PATTERN = /[\w']+/giv;
  let next;

  while ((next = PATTERN.exec(text)) != undefined) {
    n++;
  }

  return n;
}

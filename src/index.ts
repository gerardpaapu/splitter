import * as FS from 'node:fs/promises';
import {
  stringifyBody,
  stringifyHeadingIsh,
  stringifySegment,
  toInitialChunks,
} from './markdown.ts';
import * as SS from './similarity.ts';
import { wordCount } from './word-count.ts';
import type { Chunk, ChunkWithStats } from './Chunk.ts';

const TARGET_SIZE = 500;
const SIZE_FACTOR = 0.005;
const SIMILARITY_FACTOR = 200;
// TODO: I think I've got this factor weird
// what are we saying, that we *WANT* to cut at the **most** dropped
// headings or at the **fewest** retained headings
const SPLIT_HEADINGS_FACTOR = -100;
const WRITE_WINNER_TO_STDOUT = true;

interface Path {
  cuts: number[];
  error: number;
}

function howManyHeadingsDropped(previous: Chunk | undefined, current: Chunk) {
  if (previous == undefined) {
    return 0;
  }

  let matching = 0;
  for (let i = 0; i < previous.headings.length; i++) {
    let a = stringifyHeadingIsh(current.headings[i]!);
    let b = stringifyHeadingIsh(previous.headings[i]!);

    if (a !== b) {
      matching = i;
      break;
    }
  }

  return previous.headings.length - matching;
}

async function addFeatures(chunks: Chunk[]): Promise<ChunkWithStats[]> {
  console.time('calculating embeddings');

  for (const chunk of chunks) {
    chunk.embedding = await SS.extract(stringifyBody(chunk));
  }
  console.timeEnd('calculating embeddings');

  console.time('calculating similarities');
  chunks[0]!.similarity_back = 0;
  chunks[0]!.headings_dropped = 0;

  chunks.at(-1)!.similarity_forward = 0;

  for (let i = 1; i < chunks.length; i++) {
    let previous = chunks[i - 1];
    let current = chunks[i];

    let similarity = SS.similarity(
      previous?.embedding?.data as number[],
      current?.embedding?.data as number[],
    );

    previous!.similarity_forward = similarity;
    current!.similarity_back = similarity;
    current!.headings_dropped = howManyHeadingsDropped(previous, current!);
  }
  console.timeEnd('calculating similarities');
  return chunks as ChunkWithStats[];
}

async function main() {
  const txt = await FS.readFile('button.md', 'utf8');
  const incompleteChunks = toInitialChunks(txt);
  const chunks = await addFeatures(incompleteChunks);
  console.log(`chunks.length = ${chunks.length}`);
  console.time(`Searching...`);
  // solutions[i] is the best path (0..i + 1)
  const solutions = [
    { cuts: [0, 1], error: calculateError(chunks, 0, 1) },
  ] as Path[];

  for (let depth = 1; depth <= chunks.length; depth++) {
    let lowestError = Infinity;

    for (let midpoint = 1; midpoint < depth; midpoint++) {
      let m = solutions[midpoint - 1]!;
      if (m == undefined) {
        throw new Error(
          `missing earlier solutions[${midpoint - 1}] while solving for 0..${midpoint}`,
        );
      }

      let error = m.error + calculateError(chunks, midpoint, depth);
      if (error < lowestError) {
        lowestError = error;
        solutions[depth - 1] = {
          cuts: [...m.cuts, depth],
          error,
        };
      }
    }
  }
  console.timeEnd(`Searching...`);

  console.log(solutions.at(chunks.length));
  if (WRITE_WINNER_TO_STDOUT) {
    const { cuts } = solutions.at(-1)!;
    for (let i = 1; i < cuts.length; i++) {
      const start = cuts[i - 1]!;
      const end = cuts[i]!;

      const segment = chunks.slice(start, end);
      process.stdout.write(
        `\n\n### ✂️ (${start}, ${end}) dropped headings = ${chunks[start]!.headings_dropped}, similarity = ${chunks[start]!.similarity_back}✂️ ###\n\n`,
      );

      process.stdout.write(stringifySegment(segment));
    }
  }
}

function collectErrors(chunks: Chunk[], start: number, end: number) {
  let error = [];
  if (start === end) {
    error.push({
      type: 'zero-length-forbidden',
      raw: 1,
      // we just forbid these
      error: Infinity,
    });

    return error;
  }

  let segment = chunks.slice(start, end);
  let first = segment[0]!;
  let wc = wordCount(stringifySegment(segment));
  let sizeDiff = wc - TARGET_SIZE;
  error.push({
    type: 'sizediff',
    raw: sizeDiff,
    error: sizeDiff * sizeDiff * SIZE_FACTOR,
  });

  // to avoid double counting, let's count the similarity with our
  // first paragraph compared to the paragraph the previous chunk's
  // last paragraph
  let similarity = first.similarity_back ?? 0;
  error.push({
    type: 'similarity',
    raw: similarity,
    error: similarity * similarity * SIMILARITY_FACTOR,
  });

  // again, we're only evaluating the cut before this chunk
  // (should this be squared)?
  let headlinesScore = first.headings_dropped ?? 0;
  error.push({
    type: 'split-headings',
    raw: headlinesScore,
    error: headlinesScore * headlinesScore * SPLIT_HEADINGS_FACTOR,
  });

  return error;
}

function calculateError(chunks: Chunk[], start: number, end: number) {
  return collectErrors(chunks, start, end).reduce(
    (total, { error }) => total + error,
    0,
  );
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});

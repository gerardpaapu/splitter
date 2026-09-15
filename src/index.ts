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

const TARGET_WORD_COUNT = 500;

const SIZE_FACTOR = 2.5;
const SIMILARITY_FACTOR = 1;
const SPLIT_HEADINGS_FACTOR = 9_000;

const WRITE_WINNER_TO_STDOUT = true;
const FILE_PATH = process.argv[2];

if (!FILE_PATH) {
  console.error(`Missing arg FILE_PATH`);
  process.exitCode = 1;
  throw new Error('oops!')
}

interface Path {
  cuts: number[];
  error: number;
}

function howManyHeadingsDropped(previous: Chunk | undefined, current: Chunk): [number, number] {
  if (previous == undefined) {
    return [0, 0];
  }

  let matching = 0;
  for (let i = 0; i < previous.headings.length; i++) {
    let a = stringifyHeadingIsh(current.headings[i]!);
    let b = stringifyHeadingIsh(previous.headings[i]!);
    if (a !== b) {
      break;
    }

    matching++
  }

  return [matching, previous.headings.length - matching];
}

async function addFeatures(chunks: Chunk[]): Promise<ChunkWithStats[]> {
  console.time('calculating embeddings');

  for (const chunk of chunks) {
    try {
      chunk.embedding = await SS.extract(stringifyBody(chunk));
    } catch (e) {
      let text = stringifyBody(chunk);
      console.error(e);
      console.error(`embedding failed on (len = ${text.length})`);
    }
  }
  console.timeEnd('calculating embeddings');

  console.time('calculating similarities');
  chunks[0]!.similarity_back = 0;
  chunks[0]!.headings_dropped = 0;
  chunks[0]!.headings_retained = 0;


  chunks.at(-1)!.similarity_forward = 0;
  for (let i = 1; i < chunks.length; i++) {
    let previous = chunks[i - 1];
    let current = chunks[i];
    let similarity: number | undefined;
    
    similarity = SS.similarity(
      previous?.embedding!,
      current?.embedding!,
    );

    previous!.similarity_forward = similarity ?? 0;
    current!.similarity_back = similarity ?? 0;

    const [retained, dropped] = howManyHeadingsDropped(previous, current!)
    current!.headings_dropped = dropped;
    current!.headings_retained = retained;
  }
  console.timeEnd('calculating similarities');

  return chunks as ChunkWithStats[];
}

async function main() {
  console.log(`FILE_PATH=${FILE_PATH}`)
  const txt = await FS.readFile(FILE_PATH!, 'utf8');
  const incompleteChunks = toInitialChunks(txt);
  const chunks = await addFeatures(incompleteChunks);
  console.log(`chunks.length = ${chunks.length}`);
  console.time(`Searching...`);
  // solutions[i] is the best path (0..i + 1)
  const solutions = [
    { cuts: [0, 1], error: calculateError(chunks, 0, 1) },
  ] as Path[];

  for (let depth = 1; depth <= chunks.length; depth++) {
    solutions[depth - 1] = {
      cuts: [0, depth],
      error: calculateError(chunks, 0, depth)
    }

    for (let midpoint = 1; midpoint < depth; midpoint++) {
      let m = solutions[midpoint - 1]!;
      if (m == undefined) {
        throw new Error(
          `missing earlier solutions[${midpoint - 1}] while solving for 0..${midpoint}`,
        );
      }

      let error = m.error + calculateError(chunks, midpoint, depth);
      if (error < solutions[depth - 1]!.error) {
        solutions[depth - 1] = {
          cuts: [...m.cuts, depth],
          error,
        };
      }
    }
  }

  console.timeEnd(`Searching...`);

  if (WRITE_WINNER_TO_STDOUT) {
    const { cuts } = solutions.at(-1)!;
    for (let i = 1; i < cuts.length; i++) {
      const start = cuts[i - 1]!;
      const end = cuts[i]!;

      const segment = chunks.slice(start, end);
      const errors = collectErrors(chunks, start, end);
      const wc = wordCount(stringifySegment(segment));

      process.stdout.write(
        `\n\n### ✂️ (${start}) wc = ${String(wc).padStart(7)} headings split = ${chunks[start]!.headings_retained}, similarity = ${chunks[start]!.similarity_back}✂️ ###\n\n`,
      );

      process.stdout.write(stringifySegment(segment));
    }
  }
}

function collectErrors(chunks: ChunkWithStats[], start: number, end: number) {
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
  let sizeDiff = wc - TARGET_WORD_COUNT;
  first.size_diff = sizeDiff;
  error.push({
    type: 'sizediff',
    raw: sizeDiff,
    error: sizeDiff * sizeDiff * SIZE_FACTOR,
  });

  // to avoid double counting, let's count the similarity with our
  // first paragraph compared to the paragraph the previous chunk's
  // last paragraph
  let similarity = first.similarity_back;
  error.push({
    type: 'similarity',
    raw: similarity,
    error: similarity * similarity * SIMILARITY_FACTOR,
  });

  // again, we're only evaluating the cut before this chunk
  // (should this be squared)?
  let headlinesScore = first.headings_retained;
  error.push({
    type: 'split-headings',
    raw: headlinesScore,
    error: headlinesScore * headlinesScore * SPLIT_HEADINGS_FACTOR,
  });

  return error;
}

function calculateError(chunks: ChunkWithStats[], start: number, end: number) {
  const error = collectErrors(chunks, start, end).reduce(
    (total, { error }) => total + error,
    0,
  );

  if (isNaN(error)) {
    throw new Error(`invalid error calculated`);
  }

  if (!isFinite(error)) {
    throw new Error(`Infinite error = ${start}, ${end}`)
  }

  return error;
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});

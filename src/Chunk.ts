import type { Paragraph, RootContent } from 'mdast';
import type { PseudoHeading } from './markdown.ts';
import type { Tensor } from '@huggingface/transformers';

export interface Chunk {
  headings: PseudoHeading[]; // all the headings this is under closest first
  node: Paragraph | null;
  figures: RootContent[];
  embedding?: Float32Array;
  similarity_back?: number;
  similarity_forward?: number;
  headings_dropped?: number;
  headings_retained?: number;
  size_diff?: number;
}

export interface ChunkWithStats extends Chunk {
  embedding: Float32Array;
  similarity_back: number;
  similarity_forward: number;
  headings_dropped: number;
  headings_retained: number;
}

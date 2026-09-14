import type { Paragraph, RootContent } from 'mdast';
import type { PseudoHeading } from './markdown.ts';
import type { Tensor } from '@huggingface/transformers';

export interface Chunk {
  headings: PseudoHeading[]; // all the headings this is under closest first
  node: Paragraph | null;
  figures: RootContent[];
  embedding?: Tensor;
  similarity_back?: number;
  similarity_forward?: number;
  headings_dropped?: number;
}

export interface ChunkWithStats extends Chunk {
  embedding: Tensor;
  similarity_back: number;
  similarity_forward: number;
  headings_dropped: number;
}

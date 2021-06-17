export interface Experiment {
    id: number;
    name: string;
    createdOn: Date;
    progress: number;
    config: { [key: string]: number };
  }
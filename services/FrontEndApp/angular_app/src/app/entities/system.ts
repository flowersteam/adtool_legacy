export interface System {
  id: number;
  name: string;
  experiment_id: number;
  config: { [key: string]: number };
}

import { System } from '../entities/system'
import { Explorer } from '../entities/explorer'
import { InputWrapper } from '../entities/input_wrapper'
import { OutputRepresentation } from '../entities/output_representation'
import { Checkpoint } from '../entities/checkpoint'

export interface Experiment {
    id: number;
    name: string;
    created_on: Date;
    progress: number;
    exp_status: number;
    config: { [key: string]: number };
    systems: System[];
    explorers: Explorer[];
    input_wrappers: InputWrapper[];
    output_representations: OutputRepresentation[];
    checkpoints: Checkpoint[];
  }
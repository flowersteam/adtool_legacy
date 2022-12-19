import { TestBed } from '@angular/core/testing';

import { CreateNewExperimentService } from './create-new-experiment.service';

describe('CreateNewExperimentService', () => {
  let service: CreateNewExperimentService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(CreateNewExperimentService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});

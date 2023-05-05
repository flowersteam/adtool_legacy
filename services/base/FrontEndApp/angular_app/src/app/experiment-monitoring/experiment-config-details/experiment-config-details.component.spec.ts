import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ExperimentConfigDetailsComponent } from './experiment-config-details.component';

describe('ExperimentConfigDetailsComponent', () => {
  let component: ExperimentConfigDetailsComponent;
  let fixture: ComponentFixture<ExperimentConfigDetailsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ExperimentConfigDetailsComponent],
    }).compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ExperimentConfigDetailsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ExperimentMonitoringComponent } from './experiment-monitoring.component';

describe('ExperimentMonitoringComponent', () => {
  let component: ExperimentMonitoringComponent;
  let fixture: ComponentFixture<ExperimentMonitoringComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ExperimentMonitoringComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ExperimentMonitoringComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ArchiveExperimentComponent } from './archive-experiment.component';

describe('ArchiveExperimentComponent', () => {
  let component: ArchiveExperimentComponent;
  let fixture: ComponentFixture<ArchiveExperimentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ArchiveExperimentComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ArchiveExperimentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

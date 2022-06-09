import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChoosePreviousExperimentComponent } from './choose-previous-experiment.component';

describe('ChoosePreviousExperimentComponent', () => {
  let component: ChoosePreviousExperimentComponent;
  let fixture: ComponentFixture<ChoosePreviousExperimentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ChoosePreviousExperimentComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ChoosePreviousExperimentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

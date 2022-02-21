import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SetModuleConfigComponent } from './set-module-config.component';

describe('SetModuleConfigComponent', () => {
  let component: SetModuleConfigComponent;
  let fixture: ComponentFixture<SetModuleConfigComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ SetModuleConfigComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(SetModuleConfigComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

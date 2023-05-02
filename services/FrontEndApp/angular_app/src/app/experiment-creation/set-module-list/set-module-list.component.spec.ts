import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SetModuleListComponent } from './set-module-list.component';

describe('SetModuleListComponent', () => {
  let component: SetModuleListComponent;
  let fixture: ComponentFixture<SetModuleListComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [SetModuleListComponent],
    }).compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(SetModuleListComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

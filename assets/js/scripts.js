/**
 * Site-wide JS that sets up:
 *
 * [1] MathJax rendering on navigation
 * [2] Sidebar toggling
 * [3] Sidebar scroll preserving
 * [4] Keyboard navigation
 * [5] nbinteract
 */

const runWhenDOMLoaded = cb => {
  if (document.readyState != 'loading') {
    cb()
  } else if (document.addEventListener) {
    document.addEventListener('DOMContentLoaded', cb)
  } else {
    document.attachEvent('onreadystatechange', function() {
      if (document.readyState == 'complete') cb()
    })
  }
}

const togglerId = 'js-sidebar-toggle'
const textbookId = 'js-textbook'
const togglerActiveClass = 'is-active'
const textbookActiveClass = 'js-show-sidebar'

const mathRenderedClass = 'js-mathjax-rendered'

const getToggler = () => document.getElementById(togglerId)
const getTextbook = () => document.getElementById(textbookId)

// [1] Run MathJax when Turbolinks navigates to a page.
// When Turbolinks caches a page, it also saves the MathJax rendering. We mark
// each page with a CSS class after rendering to prevent double renders when
// navigating back to a cached page.
document.addEventListener('turbolinks:load', () => {
  const textbook = getTextbook()
  if (window.MathJax && !textbook.classList.contains(mathRenderedClass)) {
    MathJax.Hub.Queue(['Typeset', MathJax.Hub])
    textbook.classList.add(mathRenderedClass)
  }
})

/**
 * [2] Toggles sidebar and menu icon
 */
const toggleSidebar = () => {
  const toggler = getToggler()
  const textbook = getTextbook()

  if (textbook.classList.contains(textbookActiveClass)) {
    textbook.classList.remove(textbookActiveClass)
    toggler.classList.remove(togglerActiveClass)
  } else {
    textbook.classList.add(textbookActiveClass)
    toggler.classList.add(togglerActiveClass)
  }
}

/**
 * Keep the variable below in sync with the tablet breakpoint value in
 * _sass/inuitcss/tools/_tools.mq.scss
 *
 */
const autoCloseSidebarBreakpoint = 740

// Set up event listener for sidebar toggle button
const sidebarButtonHandler = () => {
  getToggler().addEventListener('click', toggleSidebar)

  /**
   * Auto-close sidebar on smaller screens after page load.
   *
   * Having the sidebar be open by default then closing it on page load for
   * small screens gives the illusion that the sidebar closes in response
   * to selecting a page in the sidebar. However, it does cause a bit of jank
   * on the first page load.
   *
   * Since we don't want to persist state in between page navigation, this is
   * the best we can do while optimizing for larger screens where most
   * viewers will read the textbook.
   *
   * The code below assumes that the sidebar is open by default.
   */
  if (window.innerWidth < autoCloseSidebarBreakpoint) toggleSidebar()
}

runWhenDOMLoaded(sidebarButtonHandler)
document.addEventListener('turbolinks:load', sidebarButtonHandler)

/**
 * [3] Preserve sidebar scroll when navigating between pages
 */
let sidebarScrollTop = 0
const getSidebar = () => document.getElementById('js-sidebar')

document.addEventListener('turbolinks:before-visit', () => {
  sidebarScrollTop = getSidebar().scrollTop
})

document.addEventListener('turbolinks:load', () => {
  getSidebar().scrollTop = sidebarScrollTop
})

/**
 * Focus textbook page by default so that user can scroll with spacebar
 */
const focusPage = () => {
  document.querySelector('.c-textbook__page').focus()
}
runWhenDOMLoaded(focusPage)
document.addEventListener('turbolinks:load', focusPage)

/**
 * [4] Use left and right arrow keys to navigate forward and backwards.
 */
const LEFT_ARROW_KEYCODE = 37
const RIGHT_ARROW_KEYCODE = 39

const getPrevUrl = () => document.getElementById('js-page__nav__prev').href
const getNextUrl = () => document.getElementById('js-page__nav__next').href
document.addEventListener('keydown', event => {
  const keycode = event.which

  if (keycode === LEFT_ARROW_KEYCODE) {
    Turbolinks.visit(getPrevUrl())
  } else if (keycode === RIGHT_ARROW_KEYCODE) {
    Turbolinks.visit(getNextUrl())
  }
})

/**
 * [5] Set up nbinteract to render widgets on page load
 */

let interact

const setupNbinteract = () => {
  // If NbInteract hasn't loaded, wait one second and try again
  if (window.NbInteract === undefined) {
    setTimeout(setupNbinteract, 1000)
    return
  }

  if (interact === undefined) {
    console.log('Initializing nbinteract...')
    interact = new window.NbInteract({
      baseUrl: 'https://mybinder.org',
      spec: 'DS-100/textbook/master',
      provider: 'gh',
    })
    window.interact = interact
  }

  interact.prepare()
}

runWhenDOMLoaded(setupNbinteract)
document.addEventListener('turbolinks:load', setupNbinteract)
